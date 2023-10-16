import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img
from contextlib import nullcontext
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL


class MedicalTranslationLDM(LatentDiffusion):

    def __init__(
            self,
            semantic_key,
            edge_key,
            label_key,
            semantic_beta=1.0,
            timesteps=1000,
            *args,
            **kwargs):
        super().__init__(timesteps=timesteps, *args, **kwargs)
        self.semantic_key = semantic_key
        self.edge_key = edge_key
        self.label_key = label_key
        self.semantic_threshold = timesteps - timesteps * semantic_beta

    def _get_input(self, batch, k, bs=None):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(self.device)
        x = rearrange(x, 'b h w c -> b c h w')
        if bs is not None:
            x = x[:bs]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False):

        x = self._get_input(batch, k, bs)
        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()

        if cond_key is None:
            cond_key = self.cond_stage_key

        tc = batch[cond_key]
        if not isinstance(tc[0], str):
            tc = [item.tolist() for item in tc]

        if not self.cond_stage_trainable or force_c_encode:
            if isinstance(tc, dict) or isinstance(tc, list):
                tc = self.get_learned_conditioning(tc)
            else:
                tc = self.get_learned_conditioning(tc.to(self.device))

        if bs is not None:
            tc = tc[:bs]

        if self.use_positional_encodings:
            pos_x, pos_y = self.compute_latent_shifts(batch)
            ckey = __conditioning_keys__[self.model.conditioning_key]
            tc = {ckey: tc, 'pos_x': pos_x, 'pos_y': pos_y}

        semantics = self._get_input(batch, self.semantic_key)
        sc = self.get_first_stage_encoding(self.encode_first_stage(semantics)).detach()

        edges = self._get_input(batch, self.edge_key)
        ec = self.get_first_stage_encoding(self.encode_first_stage(edges)).detach()

        lc = batch[self.label_key]

        if bs is not None:
            sc = sc[:bs]
            ec = ec[:bs]
            lc = lc[:bs]


        out = [z, dict(c_crossattn=tc, c_concat=sc, c_hint=ec, c_label=lc)]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append(tc)
        return out


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)

        c_concat = cond['c_concat']
        c_crossattn = cond['c_crossattn']
        c_hint = cond['c_hint']
        c_label = cond['c_label']

        for i, cur_t in enumerate(t):
            if cur_t < self.semantic_threshold:
                c_concat[i] = x_noisy[i]

        x = torch.cat([x_noisy] + [c_concat], dim=1)
        out = self.model.diffusion_model(x=x, hint=c_hint, label=c_label, timesteps=t, context=c_crossattn)
        return out

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        # if cond is not None:
        #     print(type(cond))
        #     if isinstance(cond, dict):
        #         cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
        #         list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
        #     else:
        #         cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates


    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=50, ddim_eta=0., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)



        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            semantics = c['c_concat']
            edges = c['c_hint']

            sc = self.decode_first_stage(semantics)
            log["conditioning_semantic"] = sc

            tc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
            log["conditioning_txt"] = tc

            ec = self.decode_first_stage(edges)
            log["conditioning_edge"] = ec

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)

                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if inpaint:
            # make a simple center square
            b, h, w = z.shape[0], z.shape[2], z.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):
                samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_inpainting"] = x_samples
            log["mask"] = mask

            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log