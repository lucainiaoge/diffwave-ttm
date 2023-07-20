# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import sys
sys.path.append("..")

import os
from pathlib import Path
from multiprocessing import cpu_count

import math

import torch
from torch.optim import Adam
from ema_pytorch import EMA
from torch.utils.data import DataLoader
import torchaudio

from accelerate import Accelerator

from tqdm.auto import tqdm
from utils import exists, cycle, num_to_groups

debug = False

class WaveDDPMTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 4,
        results_folder = './results',
    ):
        amp = False
        fp16 = False
        use_lion = False
        split_batches = True

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        
        self.diffusion_model = diffusion_model
        device = self.accelerator.device
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # optimizer

        optim_klass = Lion if use_lion else Adam
        self.opt = optim_klass(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion_model, self.opt = self.accelerator.prepare(
            self.diffusion_model, self.opt
        )
        
        # dataset and dataloader
        self.ds = dataset
        
        dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = 0) #cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        print("using device", self.accelerator.device)
        
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'diffusion_model': self.accelerator.get_state_dict(self.diffusion_model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': "test" #__version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        diffusion_model = self.accelerator.unwrap_model(self.diffusion_model)
        diffusion_model.load_state_dict(data['diffusion_model'])
        
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def sample(self, num_samples, save_dir, base_filename):
        accelerator = self.accelerator
        device = accelerator.device
        
        self.ema.ema_model.eval()

        with torch.no_grad():
            gen_data_list = [
                self.ema.ema_model.sample(batch_size=1).cpu() for n in range(num_samples)
            ]

            for i_data, audio in enumerate(gen_data_list):
                # TODO: save audio to file
                gen_data_path = os.path.join(save_dir, base_filename + "-audio-gen-{}.wav".format(i_data))
                torchaudio.save(gen_data_path, audio, self.ds.samplerate)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    audio = next(self.dl)
                    audio = audio.to(device)
                    with torch.no_grad():
                        # TODO: add spectrogram if needed
                        pass
                    
                    with self.accelerator.autocast():
                        loss = self.diffusion_model(audio)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    
                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.ema.ema_model.eval()
                        base_filename = f"sample-{milestone}"
                        self.sample(self.num_samples, self.results_folder, base_filename)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        