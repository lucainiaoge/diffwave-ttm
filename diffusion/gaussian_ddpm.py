# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import sys
sys.path.append("..")

from collections import namedtuple

import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce

from tqdm.auto import tqdm

from utils import exists, default, identity
from utils import extract, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class WaveDDPM(nn.Module):
    def __init__(
        self, 
        model, 
        signal_len,
        timesteps = 1000, 
        loss_type = 'l1', 
        objective = 'pred_x0', 
        schedule_fn_kwargs = dict()
    ):
        super().__init__()
        
        self.model = model
        self.signal_len = signal_len
        self.objective = objective

        beta_schedule_fn = sigmoid_beta_schedule
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = int(timesteps) - 1
        p2_loss_weight_gamma = 0. # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1
        self.is_ddim_sampling = True
        self.ddim_sampling_eta = 0.

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32)) # to register param for the model, this is the method of parent nn.Module

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)
        
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_output_to_prediction(self, model_output, x, t):
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            ValueError(f'unknown objective {self.objective}')
        
        return ModelPrediction(pred_noise, x_start)

    def q_posterior(self, x_start, x_t, t): #q(x_{t-1}|x_t,x_0)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None): # q(x_t|x_0)
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    
    def get_training_target(self, x_start, t, noise):
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')
        return target

    def model_predictions(self, x, t, spectrogram = None):
        model_output = self.model(x, t, spectrogram = spectrogram)
        # returning a tuple (epsilon_0|t, x_0)
        return self.model_output_to_prediction(model_output, x, t)

    def p_mean_variance(self, x, t, spectrogram = None, clip_denoised = True): # p_\theta(x_{t-1}|x_t)?
        preds = self.model_predictions(x, t, spectrogram = spectrogram) # epsilon_0|t, x_0
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start 

    @torch.no_grad()
    def p_sample(self, x, t: int, spectrogram = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x = x, 
            t = batched_times, 
            spectrogram = spectrogram, 
            clip_denoised = True
        )
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_audio = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_audio, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device

        pred_audio = torch.randn(shape, device = device)
        pred_audios = [pred_audio]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            # debug
            pred_audio, x_start = self.p_sample(pred_audio, t, spectrogram = None)
            pred_audios.append(pred_audio)

        ret = pred_audio if not return_all_timesteps else torch.stack(pred_audios, dim = 1)

        # ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        pred_audio = torch.randn(shape, device = device)
        pred_audios = [pred_audio]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            # debug
            pred_noise, x_start, *_ = self.model_predictions(pred_audio, time_cond, spectrogram = None)

            if time_next < 0:
                pred_audio = x_start
                pred_audios.append(pred_audio)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(pred_audio)

            pred_audio = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            pred_audios.append(pred_audio)

        ret = pred_audio if not return_all_timesteps else torch.stack(pred_audios, dim = 1)

        # ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        signal_len = self.signal_len
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            (batch_size, signal_len), 
            return_all_timesteps = return_all_timesteps
        )

    def p_losses(self, x_start, t, noise = None):
    # given x0, calculate ||x0 - f_\theta(a' x0 + b' eta)|| (for x0 predictive net)
        b, l = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model(x, t, spectrogram = None) # TODO: add load spectrogram
        target = self.get_training_target(x_start, t, noise)

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, audio, *args, **kwargs):
        b, l, device, signal_len, = *audio.shape, audio.device, self.signal_len
        assert l == signal_len, f'length of signal must be {signal_len}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # audio = self.normalize(audio)
        return self.p_losses(audio, t, *args, **kwargs)


