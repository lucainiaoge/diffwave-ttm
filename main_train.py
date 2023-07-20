import numpy as np
from diffusion.model_diffusion import DiffWaveNet
from diffusion.gaussian_ddpm import WaveDDPM
from data.chunked_audio_dataset import ChunkedAudioDataset
from diffusion.train_ddpm import WaveDDPMTrainer

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

params = AttrDict(
    # Training params
    batch_size=16, # 16
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate = 4000, # 22050,
    start_silence_sec=3.0,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Probably an error in paper.

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional = True,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # unconditional sample len
    audio_len = 32000, # unconditional_synthesis_samples
)


model_diffwave = DiffWaveNet(params)

wave_ddpm = WaveDDPM(
    model_diffwave,
    params.audio_len,
    timesteps = len(params.noise_schedule),   # number of sampling steps
)

data_folder = "DougMcKenzie"
dataset = ChunkedAudioDataset(
    data_folder, 
    params.audio_len, 
    samplerate = params.sample_rate, 
    start_silence_sec = params.start_silence_sec
)

trainer = WaveDDPMTrainer(
    wave_ddpm, 
    dataset, 
    train_batch_size = params.batch_size, 
    train_lr = params.learning_rate,
    train_num_steps = 1000000,
    save_and_sample_every = 5000,
    num_samples = 4,
    results_folder = './results',
)

trainer.train()
