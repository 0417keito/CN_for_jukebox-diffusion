from typing import Callable, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
from flat_audio_diffusion.apex import XUNet, DownsampleItem
from a_unet.blocks import Sequential, default
from typing import Callable, Optional, Sequence
import torch as t
import numpy as np
from torch import Tensor
from jukebox.utils.io import load_audio
from einops import rearrange, repeat
# t.set_float32_matmul_precision('medium')
from audio_diffusion_pytorch.diffusion import VSampler, Diffusion, Distribution, UniformDistribution
import tqdm
from jbdiff.utils import batch_postprocess, batch_preprocess, extend_dim
from audio_diffusion_pytorch.utils import groupby
from math import pi
from typing import Any, Optional, Tuple
import torch.nn.functional as F
from a_unet.blocks import T
import pytorch_lightning as pl

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ControlledUnetModel(XUNet):
    def forward(
        self,
        x: Tensor,
        *,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
        **kwargs
    ) -> Tensor:
        controls = kwargs['control_out']
        skips = []
        for down, skip_adapter, in zip(self.all_down, self.all_skip_adapter):
            skip = skip_adapter(x)
            skips.append(skip)
            x = down(x, features, embedding, channels)

        for up, skip in zip(self.all_up, reversed(self.all_skip)):
            x = up(x, features, embedding, channels)
            x = skip(skips.pop()+controls.pop(), x, features)
        
        return x
    
    
class Control_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_t: Callable = DownsampleItem,
        items: Sequence[Callable] = [],
        out_channels: Optional[int] = None,
        resnet_dilation_factor: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        items_down = [downsample_t] + list(items)
        items_kwargs = dict(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
        # Build items stack: items down -> inner block -> items up
        exp = list(reversed(range(99999)))
        items_down = [item_t(dilation=resnet_dilation_factor**exp.pop(), **items_kwargs) if item_t.__name__ == 'ResnetItem' else item_t(**items_kwargs) for item_t in items_down]
        self.items_down = Sequential(*items_down)
        self.zero_conv = zero_module(downsample_t(dim=1, factor=1, in_channels=in_channels, channels=in_channels))

Control_XBlock = T(Control_Block, override=False)

class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        blocks: Sequence,
        out_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        num_layers = len(blocks)
        out_channels = default(out_channels, in_channels)
        
        nets = []
        for i in range(num_layers):
            in_ch = in_channels if i==0 else blocks[i-1].channels
            out_ch = out_channels if i==0 else in_ch
            block_t = blocks[i] 
            nets.append(block_t(
                in_channels=in_ch,
                out_channels=out_ch,
                depth=i,
                **kwargs
            ))
        all_down = []
        all_zero_convs = []
        for net in nets:
            all_down.append(net.items_down)
            all_zero_convs.append(net.zero_conv)
        self.all_down = nn.ModuleList(all_down)
        
        self.input_hint_block = []
        
    def forward(
        self,
        x: Tensor,
        *,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
        guided_hint: Optional[Tensor] = None,
    ) -> Tensor:
        #guided_hint = self.input_hint_block(hint, features, embedding, channels)
        outs = []
        for down, zero_conv in zip(self.all_down, self.zero_convs):
            if guided_hint is not None:
                out = zero_conv(x)
                outs.append(out)
                x = down(x, features, embedding, channels)
                x += guided_hint
                guided_hint = None
            else:
                out = zero_conv(x)
                outs.append(out)
                x = down(x, features, embedding, channels)
            
        return outs
    
class Control_VDiffusion(Diffusion):
    def __init__(
        self, net: nn.Module, controlnet:nn.Module, sigma_distribution: Distribution = UniformDistribution(), loss_fn: Any = F.mse_loss
    ):
        super().__init__()
        self.net = net
        self.controlnet = controlnet
        self.sigma_distribution = sigma_distribution
        self.loss_fn = loss_fn

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        if noise is None:
            noise = torch.randn_like(x)
        #print('noise: ', noise, '\nmean: ', noise.mean(), '\nstd: ', noise.std())
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        control_out = self.controlnet(x_noisy, sigmas, **kwargs)
        kwargs['control_out'] = control_out
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return self.loss_fn(v_pred, v_target)
    
class Control_DiffusionModel(nn.Module):
    def __init__(
        self,
        net_t: Callable,
        controlnet_t: Callable,
        diffusion_t: Callable = Control_VDiffusion,
        sampler_t: Callable = VSampler,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        sampler_kwargs, kwargs = groupby("sampler_", kwargs)

        self.net = net_t(dim=dim, **kwargs)
        self.controlnet = controlnet_t(dim=dim, **kwargs)
        self.diffusion = diffusion_t(net=self.net, controlnet=self.controlnet, loss_fn=loss_fn, **diffusion_kwargs)
        self.sampler = sampler_t(net=self.net, **sampler_kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.diffusion(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs) -> Tensor:
        return self.sampler(*args, **kwargs)
    
    
class Control_JBDiffusion(pl.LightningModule):
    '''
    JBDiffusion class to be trained

    Init Params
    ___________
    - vqvae: (Jukebox) instance of the vqvae encoder/decoder for retrieving latent codes
    - level: (int) level of vqvae latent codes to train on (2->0)
    - diffusion_kwargs: (dict) dict of diffusion kwargs
    '''
    def __init__(self, vqvae, level, diffusion_kwargs):
        super().__init__()

        self.level = level
        self.upsampler = self.level in (0,1)
        self.diffusion = Control_DiffusionModel(**diffusion_kwargs)
        # self.diffusion_ema = deepcopy(self.diffusion)
        # self.ema_decay = global_args.ema_decay
        self.vqvae = vqvae
    
    def configure_optimizers(self):
        lr = 4e-5
        params = list(self.diffusion.controlnet_t.parameters())
        opt = t.optim.Adam(params, lr=lr)
        return opt

    def training_step(self, batch, batch_idx):
        # Assure training
        self.diffusion.train()
        assert self.diffusion.training
        # Grab train batch
        x, cond, control = batch
        # Preprocess batch and conditional audio for diffusion (includes running batch through Jukebox encoder)
        z_q, x_q = batch_preprocess(x, self.vqvae, self.level)
        control_z_q, control_x_q = batch_preprocess(control, self.vqvae, self.level)
        cond_z, cond_q = batch_preprocess(cond, self.vqvae, self.level)
        cond_q = rearrange(cond_q, "b c t -> b t c")
        # Upsampler uses noisy data from level below
        if self.upsampler:
            # Run example through level below back to noisy audio
            _, x_noise_q = batch_preprocess(x, self.vqvae, self.level+1)
            x_noise_audio, _, _ = batch_postprocess(x_noise_q, self.vqvae, self.level+1)
            x_noise_audio = rearrange(x_noise_audio, "b c t -> b t c")
            # Preprocess and encode noisy audio at current level
            _, x_noise = batch_preprocess(x_noise_audio, self.vqvae, self.level)
            with t.cuda.amp.autocast():
                # Step
                loss = self.diffusion(x_q, noise=x_noise, embedding=cond_q, guided_hint=control_x_q, embedding_mask_proba=0.1) 
        else:
            with t.cuda.amp.autocast():
                # Step
                loss = self.diffusion(x_q, embedding=cond_q, guided_hint=control_x_q, embedding_mask_proba=0.1)

        log_dict = {
            'train/loss': loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def sample(self, noise, num_steps, init, init_strength, context, context_strength):
        if init is not None:
            start_step = int(init_strength*num_steps)
            sigmas = self.diffusion.sampler.schedule(num_steps + 1, device='cuda')
            sigmas = sigmas[start_step:]
            sigmas = repeat(sigmas, "i -> i b", b=1)
            sigmas_batch = extend_dim(sigmas, dim=noise.ndim + 1)
            alphas, betas = self.diffusion.sampler.get_alpha_beta(sigmas_batch)
            alpha, beta = alphas[0], betas[0]
            x_noisy = alpha*init + beta*noise
            progress_bar = tqdm.tqdm(range(num_steps-start_step), disable=False)

            for i in progress_bar:
                v_pred = self.diffusion.sampler.net(x_noisy, sigmas[i], embedding=context, embedding_scale=context_strength)
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
                progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

            x_noisy_audio, _, _= batch_postprocess(x_noisy, self.vqvae, self.level)
            x_noisy_audio = rearrange(x_noisy_audio, "b c t -> b t c")

            return x_noisy, x_noisy_audio
        else:
            sample = self.diffusion.sample(
                    noise,
                    embedding=context,
                    embedding_scale=context_strength, 
                    num_steps=num_steps
                    )

            sample_audio, _, _= batch_postprocess(sample, self.vqvae, self.level)
            sample_audio = rearrange(sample_audio, "b c t -> b t c")

            return sample, sample_audio

    def get_init_context(self, context_audio_file, level_mults, context_num_frames, base_tokens, context_mult, sr):
        level_mult = level_mults[self.level]
        context_frames = context_mult*base_tokens*level_mult
        cutoff = context_frames if context_frames <= context_num_frames else context_num_frames
        offset = max(0, int(context_num_frames-context_frames))
        if context_audio_file is not None:
            data, _ = load_audio(context_audio_file, sr=sr, offset=offset, duration=cutoff)
        else:
            data = np.zeros((2, context_frames))
        context = np.zeros((data.shape[0], context_frames))
        context[:, -cutoff:] += data
        context = context.T
        context = t.tensor(np.expand_dims(context, axis=0)).to('cuda', non_blocking=True).detach()
        context_z, context_q = batch_preprocess(context, self.vqvae, self.level)
        context_q = rearrange(context_q, "b c t -> b t c")

        return context_q

    def encode(self, audio):
        return batch_preprocess(audio, self.vqvae, self.level)

    def decode(self, audio_q):
        decoded_audio, _, _ = batch_postprocess(audio_q, self.vqvae, self.level)
        return rearrange(decoded_audio, "b c t -> b t c")