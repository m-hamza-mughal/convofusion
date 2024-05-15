import inspect
import os
# from convofusion.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from convofusion.config import instantiate_from_config
from os.path import join as pjoin
from convofusion.models.architectures import (
    denoiser,
    vae,
)
from convofusion.models.losses.diffvae import Losses
from convofusion.models.modeltype.base import BaseModel
from convofusion.utils.temos_utils import remove_padding
from convofusion.models.architectures.audioenc import TextAudioController
from convofusion.models.architectures.condfuser import TextAudioMotionFuser
import convofusion.models.tools.word_excitation_guidance as weg

from nltk.tokenize import word_tokenize
import nltk
import random
import math

import torch.nn.functional as F
from kornia.filters.kernels import laplacian_1d

from .base import BaseModel

import time


class Convofusion(BaseModel):
    """
    This model class is built for the ConvoFusion model.
    Based on components contained in MLD repository. 

    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncondp = cfg.model.guidance_uncondp
        self.clf_guidance_drops = 6
        self.datamodule = datamodule
        self.WEG_type = cfg.TRAIN.ABLATION.WEG_TYPE
        
        self.weg_parameters = cfg.model.weg_parameters


        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")
        
        
        if self.vae_type != "no":
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type == "convofusion":
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        if self.stage in ["diffusion", "vae_diffusion"]:
            # breakpoint()
            self.text_audio_encoder = TextAudioController(cfg, cfg.denoiser.params.text_encoded_dim)
            self.condition_fuser = TextAudioMotionFuser(cfg, cfg.denoiser.params.text_encoded_dim)
            if self.vae_type == "no":
                self.text_audio_encoder = TextAudioController(cfg, cfg.DATASET.SAMPLER.MAX_LEN)
                self.condition_fuser = TextAudioMotionFuser(cfg, self.latent_dim[-1])
            self.denoiser = instantiate_from_config(cfg.model.denoiser)
            if not self.predict_epsilon:
                cfg.model.scheduler.params['prediction_type'] = 'sample'
                cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
            self.scheduler = instantiate_from_config(cfg.model.scheduler)
            self.noise_scheduler = instantiate_from_config(
                cfg.model.noise_scheduler)

        # if self.condition in ["text", "text_uncond"]:
        #     self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "convofusion":
            self._losses = MetricCollection({
                split: Losses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        # if self.condition in ['text', 'text_uncond']:
        #     self.feats2joints = datamodule.feats2joints
        # elif self.condition == 'action':
        #     pass
        # elif self.condition == 'text+audio':
        #     pass
        #     # self.feats2joints = self.datamodule.rep6d2joints


        # laplace kernel
        # breakpoint()
        self.laplace_kernel_size = cfg.model.motion_vae.params.laplace_kernel_size
        if self.laplace_kernel_size > 0:
            self.laplace_kernel = laplacian_1d(self.laplace_kernel_size)[None, None, :]
            self.laplace_kernel.requires_grad = False
    

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        
        lengths = batch["length"]
        if self.stage in ['diffusion', 'vae_diffusion']:
            text_lsn = batch["text_lsn"]
            text_spk = batch["text_spk"]
            # audio = batch["audio"]
            melspec_spk = batch["melspec_spk"]
            melspec_lsn = batch["melspec_lsn"]
            active_passive_bit = batch["active_passive_lsn"]
            motion_spk = batch["motion_spk"]
            lsn_id = batch["lsn_id"]


        # TODO: Generate text and audio betas for controlling (default: 0.5 for each)
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            if self.condition in ['text', 'text_uncond']:
                cond = texts    
            elif self.condition == 'text+audio':
                text_cond = [text_spk, text_lsn]
                audio_cond = [melspec_spk, melspec_lsn]
            else:
                raise NotImplementedError
            # diffusion reverse
            if self.do_classifier_free_guidance: # modality guidance # TODO: Change this function
                uncond_tokens = ['-'*10] * len(texts) # uncond_tokens: (batch_size,)
                if self.condition == 'text':
                    uncond_tokens.extend(texts) # uncond_tokens (bs*2, ) 
                    cond = uncond_tokens
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                    cond = uncond_tokens
                elif self.condition == 'text+audio':
                    text_cond = [['-'*10] * len(texts) + texts for texts in [text_spk, text_lsn]] # uncond_tokens + text (bs*2, )
                    uncond_mel = -90 * torch.ones_like(melspec_lsn)
                    uncond_mel[..., 40:45] = 0
                    audio_cond = [torch.cat([uncond_mel, melspec], dim=0) for melspec in [melspec_spk, melspec_lsn]]# audio (bs*2, 128, 80)
                     # (bs*2, 512)
                    motion_spk_cond = torch.cat([torch.zeros_like(motion_spk), motion_spk], dim=0)
                    # breakpoint()
                    active_passive_bit = torch.cat([2*torch.ones_like(active_passive_bit), active_passive_bit], dim=0)
                    lsn_id = [0] * len(lsn_id) + lsn_id
                    
                else:
                    raise NotImplementedError
                
            if self.condition in ['text', 'text_uncond']:
                cond_emb = self.text_encoder(cond)
            elif self.condition == 'text+audio':
                # breakpoint()
                # text_spk_cond = text_cond[0]
                text_lsn_cond = text_cond[1]
                # melspec_spk_cond = audio_cond[0]
                melspec_lsn_cond = audio_cond[1]

                # aspk, tspk = self.text_audio_encoder(text_spk_cond, melspec_spk_cond, person_type='spk')
                alsn, tlsn, al_mask, tl_mask = self.text_audio_encoder(text_lsn_cond, melspec_lsn_cond, person_type='lsn')

                e_lengths = lengths * 2

                if self.vae_type == "no":
                    motion_spk_emb = motion_spk_cond.permute(2, 0, 1)
                else:
                    motion_spk_emb, dist_spk, _ = self.vae.encode(motion_spk_cond, e_lengths)
                    motion_spk_emb = motion_spk_emb.permute(1, 2, 0, 3)
                # motion_spk_emb = motion_spk_emb.permute(1, 2, 0)

                cond_emb = self.condition_fuser(motion_spk_emb, alsn, tlsn, active_passive_bit, lsn_id)
            else:
                raise NotImplementedError
            
            z = self._diffusion_reverse(cond_emb, lengths, cond_masks={'alsn': al_mask, 'tlsn': tl_mask})
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m, _ = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type == "convofusion":
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            # self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        # joints = self.feats2joints(feats_rst.detach().cpu())
        joints = feats_rst.detach().cpu()
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        joints = feats_rst.detach().cpu()
        return joints
    
    
    
    def iterative_refinement_step(self,
                                  latents: torch.Tensor,
                                    indices_to_alter: list[int],
                                    loss: torch.Tensor,
                                    threshold: float,
                                    encoder_hidden_states: torch.Tensor,
                                    lengths: list[int],
                                    cond_masks: dict,

                                    step_size: float,
                                    t: int,

                                    smooth_attentions: bool = True,
                                    sigma: float = 0.5,
                                    kernel_size: int = 3,
                                    max_refinement_steps: int = 400,
                                    normalize_eot: bool = False,
                                    eot_indices=[]):
        """
        Performs the iterative latent refinement introduced in Attend-and-Excite Paper.
        https://github.com/yuval-alaluf/Attend-and-Excite
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)

            # breakpoint()
            noise_pred_text, text_only_att_mats = self.denoiser(
                sample=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                mem_mask_dict=cond_masks
            ) #[0]
            self.denoiser.zero_grad()

            text_att_mats = text_only_att_mats[2]

                # aggregate and Get max activation value for each focus token defined by focus indices
            text_att_mats = weg.aggregate_attentions(text_att_mats)
            max_attention_at_indices = weg.get_max_attention_at_indices(text_att_mats, indices_to_alter, smooth_attentions=True, normalize_eot=normalize_eot, eot_indices=eot_indices)

            loss, losses = weg.compute_attention_focus_loss(max_attention_at_indices)
            if loss.all() != 0:
                # breakpoint()
                latents = weg.update_latent(latents=latents, loss=loss, lr=step_size)

            # with torch.no_grad():
            #     noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
            #     noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            # try:
            #     low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            # except Exception as e:
            #     print(e)  # catch edge case :)
            #     low_token = np.argmax(losses)

            # low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            # print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {1. - loss.item()}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)

        # breakpoint()
        noise_pred_text, text_only_att_mats = self.denoiser(
            sample=latents,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            mem_mask_dict=cond_masks
        ) #[0]
        self.denoiser.zero_grad()

        text_att_mats = text_only_att_mats[2]

            # aggregate and Get max activation value for each focus token defined by focus indices
        text_att_mats = weg.aggregate_attentions(text_att_mats)
        max_attention_at_indices = weg.get_max_attention_at_indices(text_att_mats, indices_to_alter, smooth_attentions=True, normalize_eot=normalize_eot, eot_indices=eot_indices)

        loss, losses = weg.compute_attention_focus_loss(max_attention_at_indices)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_at_indices
    

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, cond_masks=dict(), focus_indices=[]):
        # init latents
        # breakpoint()
        # WEG parameters
        scale_range = self.weg_parameters['scale_range']
        thresholds = self.weg_parameters['thresholds']

        bsz = encoder_hidden_states[0].shape[0]
        if self.do_classifier_free_guidance:
            guidance_bs_mulitplier = self.clf_guidance_drops + 1
            bsz = bsz // guidance_bs_mulitplier
        
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion" # this is not latent diffusion its simple diffusion
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states[0].device,
                dtype=torch.float,
            )
        else:
            
            latents = torch.randn(
                (bsz, 16, self.latent_dim[-1]), # 8 num_chunks x 2 for body and hands 
                device=encoder_hidden_states[0].device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        
        attention_matrices = dict()
        # breakpoint()
        for i, t in enumerate(timesteps):
            
            if len(focus_indices) > 0:
                with torch.inference_mode(False):
                    # breakpoint()
                    

                    scale_range = np.linspace(scale_range[0], 
                                              scale_range[1], 
                                              len(self.scheduler.timesteps))

                    # 
                    latents = latents.clone().detach().requires_grad_(True)

                    text_only_encoder_hidden_states = [ enc.chunk(guidance_bs_mulitplier)[1] for enc in encoder_hidden_states]
                    text_only_cond_masks = {k: v.chunk(guidance_bs_mulitplier)[1] if v is not None else v for k, v in cond_masks.items()}
                    # breakpoint()
                    noise_pred_text, text_only_att_mats = self.denoiser(
                        sample=latents,
                        timestep=t,
                        encoder_hidden_states=text_only_encoder_hidden_states,
                        lengths=lengths,
                        mem_mask_dict=text_only_cond_masks
                    ) #[0]
                    self.denoiser.zero_grad()
                    # breakpoint()
                    eot_indices = torch.argmax(text_only_cond_masks['tlsn'].int(),  dim=1) - 1

                    # text_only_att_mats = [att_mat.chunk(guidance_bs_mulitplier)[1] for att_mat in att_mats]
                    # check the shapes of the attention matrices
                    text_att_mats = text_only_att_mats[2]

                    # aggregate and Get max activation value for each focus token defined by focus indices
                    text_att_mats = weg.aggregate_attentions(text_att_mats)
                    max_attention_at_indices = weg.get_max_attention_at_indices(text_att_mats, focus_indices, smooth_attentions=True, normalize_eot=True, eot_indices=eot_indices)
                    
                    loss, losses = weg.compute_attention_focus_loss(max_attention_at_indices)

                    # TODO: add iterative refinement step here after testing
                    if i in thresholds.keys() and loss > 1. - thresholds[i]:
                        del noise_pred_text
                        torch.cuda.empty_cache()
                        loss, latents, max_attention_at_indices = self.iterative_refinement_step(latents=latents,
                                        indices_to_alter=focus_indices,
                                        loss=loss,
                                        threshold=thresholds[i],
                                        encoder_hidden_states=text_only_encoder_hidden_states,
                                        lengths=lengths,
                                        cond_masks=text_only_cond_masks,
                                        step_size=self.weg_parameters['scale_factor'] * np.sqrt(scale_range[i]),
                                        t=t,
                                        max_refinement_steps=self.weg_parameters['max_refinement_steps'],
                                        normalize_eot=True, eot_indices=eot_indices)
                        
                    
                    if i < self.weg_parameters['max_iter_to_alter']:
                        # compute loss w.r.t latent
                        loss, losses = weg.compute_attention_focus_loss(max_attention_at_indices)
                        if loss.all() != 0:
                            # breakpoint()
                            latents = weg.update_latent(latents=latents, loss=loss, lr=self.weg_parameters['scale_factor'] * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss:0.4f}')
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                guidance_bs_mulitplier) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * guidance_bs_mulitplier if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            # breakpoint()
            noise_pred, att_mats = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
                mem_mask_dict=cond_masks
            ) #[0]
            # breakpoint()
            if len(focus_indices) > 0: # for text only case
                # att_mats = text_only_att_mats # [att_mat.chunk(guidance_bs_mulitplier)[1] for att_mat in att_mats]
                att_mats = [att_mat.chunk(guidance_bs_mulitplier)[-1] for att_mat in att_mats] 
            else:
                att_mats = [att_mat.chunk(guidance_bs_mulitplier)[-1] for att_mat in att_mats] 

            # att_mats = [att_mat.chunk(guidance_bs_mulitplier)[-1] for att_mat in att_mats] 
            
            attention_matrices[t.item()] = att_mats


            # perform guidance
            if self.do_classifier_free_guidance:
                # drop sequence: all_drop, text_drop, audio_drop, spk_drop, apb_drop, lsnid_drop, no_drop
                noise_pred_uncond, noise_pred_textonly, noise_pred_audioonly, noise_pred_spkonly, \
                    noise_pred_apbonly, noise_pred_lsnidonly, noise_pred_fullcond = noise_pred.chunk(guidance_bs_mulitplier)
                
                # here modality guidance is performed
                # w_c for each modality is kept 1 as a standard. change and experiment if needed
                noise_pred_text = self.guidance_scale * 1 * (noise_pred_textonly - noise_pred_uncond)
                noise_pred_audio = self.guidance_scale * 1 * (noise_pred_audioonly - noise_pred_uncond)
                noise_pred_spk = self.guidance_scale * 1 * (noise_pred_spkonly - noise_pred_uncond)
                noise_pred_apb = self.guidance_scale  * 1 * (noise_pred_apbonly - noise_pred_uncond)
                noise_pred_lsnid = self.guidance_scale  * 1 * (noise_pred_lsnidonly - noise_pred_uncond)
                noise_pred_all = self.guidance_scale * 0 * (noise_pred_fullcond - noise_pred_uncond) 
                
                noise_pred =  noise_pred_uncond + (noise_pred_text + noise_pred_audio + noise_pred_spk + noise_pred_apb + noise_pred_lsnid + noise_pred_all)
            
            # att_mats = [att_mat.chunk(guidance_bs_mulitplier)[1] for att_mat in att_mats]
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            

        latents = latents.permute(1, 0, 2)
        return latents, attention_matrices #att_mats
    

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None, drop_idxs=None, cond_masks=dict()):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # breakpoint()
        bs, t, bh, dim = latents.shape
        latents = latents.reshape(bs, t * bh, dim) # -> bs, 8*2, dim 

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise, timesteps)
        # 
        # Predict the noise residual
        noise_pred, att_mats = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
            mem_mask_dict=cond_masks
        ) #[0]
        
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0

        if self.cfg.LOSS.LAMBDA_LATENT != 0.0:
            if drop_idxs is None:
                drop_idxs = [False] * len(noise_pred)

            # breakpoint()
            noise_pred_cond = [n for i, n in enumerate(noise_pred) if not drop_idxs[i]]
            noise_pred_cond = torch.stack(noise_pred_cond)
            noisy_latents_cond = [n for i, n in enumerate(noisy_latents) if not drop_idxs[i]]
            noisy_latents_cond = torch.stack(noisy_latents_cond)
            time_steps_cond = [n for i, n in enumerate(timesteps) if not drop_idxs[i]]
            time_steps_cond = torch.stack(time_steps_cond)

            # print(time_steps_cond.shape)
            
            extra_step_kwargs = {}
            if "eta" in set(
                    inspect.signature(self.scheduler.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
            
            pred_latents = []
            for idx, t in enumerate(time_steps_cond):
                pred_x0 = self.scheduler.step(noise_pred_cond[idx], t, noisy_latents_cond[idx], **extra_step_kwargs).pred_original_sample
                pred_latents.append(pred_x0)
            pred_latents = torch.stack(pred_latents)
            gt_latents = torch.stack([n for i, n in enumerate(latents) if not drop_idxs[i]])
            latloss_weights = self.scheduler.betas[time_steps_cond.to('cpu')].to(pred_latents.device) # HARD CODED device
        else:
            pred_latents = 0
            gt_latents = 0
            latloss_weights = 0

        # breakpoint()
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
            "train_attention_maps": att_mats,
            "lat_t": pred_latents,
            "lat_gt": gt_latents,
            "latloss_weights": latloss_weights,

        }
        # print(self.predict_epsilon)
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        # print(feats_ref.shape)

        if self.vae_type == "convofusion":
            # breakpoint()
            n_chunks = 8
            chunk_len = feats_ref.shape[1] // n_chunks

            motion_z, dist_m, _ = self.vae.encode(feats_ref, lengths)
            # breakpoint()
            pred_motion = self.vae.decode(motion_z, lengths)
            
            feats_rst = pred_motion.clone()
            
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm, _ = self.vae.encode(feats_rst, lengths)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], pred_motion.shape[1])
        # 
        # apply laplace over the motion for Laplace loss
        if self.laplace_kernel_size > 0:
            bs, t, f = feats_ref.shape
            self.laplace_kernel = self.laplace_kernel.to(feats_ref.device)
            ref_vec = feats_ref.permute(0, 2, 1).reshape(-1, 1, feats_ref.shape[1])
            rst_vec = pred_motion.permute(0, 2, 1).reshape(-1, 1, pred_motion.shape[1])
            laplace_ref = F.conv1d(ref_vec, self.laplace_kernel).reshape(
                bs, self.nfeats, -1).permute(0, 2, 1)
            laplace_rst = F.conv1d(rst_vec, self.laplace_kernel).reshape(
                bs, self.nfeats, -1).permute(0, 2, 1)
        else:
            laplace_ref = torch.zeros_like(feats_ref)
            laplace_rst = torch.zeros_like(pred_motion)

        
        lat_m = motion_z.permute(1, 2, 0, 3) 

        lat_rm = recons_z.permute(1, 2, 0, 3) 
        
        # 
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :], #
            "m_rst": pred_motion[:, :min_len, :],
            "m_laplace_ref": laplace_ref,
            "m_laplace_rst": laplace_rst,

            "lat_m": lat_m,
            "lat_rm": lat_rm,
            
            "dist_m": dist_m,
            "dist_ref": dist_ref,
            "pred_motion": feats_rst[:, :min_len, :],
            "gt_motion": feats_ref[:, :min_len, :]
        }
        return rs_set

    def train_diffusion_forward(self, batch):
        # breakpoint()
        feats_ref = batch["motion_lsn"]
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type == "convofusion":
                z, dist, _ = self.vae.encode(feats_ref, lengths)
                z = z.permute(1, 2, 0, 3) # bh, bs, t, dim -> bs, t, bh, dim

            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        
        if self.condition == 'text+audio':
            # 
            text_lsn = batch["text_lsn"].copy()
            text_spk = batch["text_spk"].copy()
            melspec_spk = batch["melspec_spk"].clone()
            melspec_lsn = batch["melspec_lsn"].clone()
            active_passive_bit = batch["active_passive_lsn"].clone()
            motion_spk = batch["motion_spk"]
            lsn_id = batch["lsn_id"]
            
            # modality guidance: randomly drop modalities during training
            # 
            num_guidance_drops = self.clf_guidance_drops
            drop_idxs = np.array_split(
                np.random.choice(a=len(text_lsn),
                                 size=int(self.guidance_uncondp*len(text_lsn))*num_guidance_drops,
                                 replace=False),
                num_guidance_drops
            )
            all_drop, text_drop, audio_drop, spk_drop, apb_drop, lsnid_drop = drop_idxs

            # all_drop except text idxs
            for idx in np.concatenate([all_drop, audio_drop, spk_drop, apb_drop, lsnid_drop]):
                text_lsn[idx] = '-'*10
            
            # all drop except audio idxs
            uncond_mel = -90 * torch.ones_like(melspec_lsn[0, :, :])
            uncond_mel[..., 40:45] = 0

            for idx in np.concatenate([all_drop, text_drop, spk_drop, apb_drop, lsnid_drop]):
                melspec_lsn[idx] = uncond_mel

            # all drop except spk idxs
            for idx in np.concatenate([all_drop, text_drop, audio_drop, apb_drop, lsnid_drop]):
                text_spk[idx] = '-'*10

            for idx in np.concatenate([all_drop, text_drop, audio_drop, apb_drop, lsnid_drop]):
                melspec_spk[idx] = uncond_mel
            
            # all drop except apb idxs
            uncond_apb = 2*torch.ones_like(active_passive_bit[0])
            for idx in np.concatenate([all_drop, text_drop, audio_drop, spk_drop, lsnid_drop]):
                active_passive_bit[idx] = uncond_apb

            # all drop except lsnid idxs
            for idx in np.concatenate([all_drop, text_drop, audio_drop, spk_drop, apb_drop]):
                lsn_id[idx] = 0
            
            
            # text audio encode
            # breakpoint()
            # aspk, tspk, as_mask, ts_mask,token2word_map_spk,  ta_spk = self.text_audio_encoder(text_spk, melspec_spk, person_type='spk-ta')
            aspk, tspk, as_mask, ts_mask, token2word_map_spk, _ = self.text_audio_encoder(text_spk, melspec_spk, person_type='spk', return_textmap=False)
            alsn, tlsn, al_mask, tl_mask, token2word_map_lsn, _ = self.text_audio_encoder(text_lsn, melspec_lsn, person_type='lsn', return_textmap=False)

            # motion encode
            # if self.vae_type == "no":
            #     motion_spk_emb = motion_spk.permute(2, 0, 1)
            # else:
            #     motion_spk_emb, dist_spk, _ = self.vae.encode(motion_spk, lengths)
            #     motion_spk_emb = motion_spk_emb.permute(1, 2, 0, 3)
                
            # motion_spk_emb = motion_spk_emb.permute(1, 2, 0)

            # when motion is used as spkemb
            # spk_emb = motion_spk_emb
            # when spk text and audio are used as spkemb
            # spk_emb = ta_spk
            # when only spk text is used as spkemb
            spk_emb = tspk

            cond_emb = self.condition_fuser(spk_emb, alsn, tlsn, active_passive_bit, lsn_id)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        # breakpoint()
        # z_timeavg = (z[:, 1:, :, :] + z[:, :-1, :, :]) / 2 # HARD CODED - changed for ikhsi's idea
        # z = torch.cat([z, z_timeavg], dim=1) # HARD CODED - changed for ikhsi's idea
        
        n_set = self._diffusion_process(z, cond_emb, lengths, drop_idxs=None, cond_masks={'alsn': al_mask, 'tlsn': tl_mask, 'spkemb':ts_mask})
        # n_set['dist_m1'] = dist
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False, split="test"):
        # breakpoint()
        # breakpoint()
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = ['-'*10] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        elif self.condition == 'text+audio':
            # REACT CHANGE
            # breakpoint()
            text_lsn = batch["text_lsn"].copy()
            text_spk = batch["text_spk"].copy()
            # audio = batch["audio"]
            melspec_spk = batch["melspec_spk"].clone()
            melspec_lsn = batch["melspec_lsn"].clone()
            active_passive_bit = batch["active_passive_lsn"].clone()
            motion_spk = batch["motion_spk"].clone()

            lsn_id = batch["lsn_id"]

            # breakpoint()
            # Remove classifier free guidance in testing phase
            full_text_lsn = text_lsn.copy()

            bs = len(text_lsn)

            # breakpoint()


            # RANDOMLY SELECT FOCUS WORDS
            
            if self.WEG_type == 'semantic':
                # utilize semantic information to select focus words from BEAT dataset 
                # breakpoint()
                assert self.datamodule._sample_set.dataset_select == 'beat', "Semantic WEG only supported for BEAT dataset"
                focus_words = [[entry['word'] for entry in batch['sem_info'][i] if isinstance(entry['word'], str)] for i in range(bs)]
            elif self.WEG_type == 'random':
                text_tokenized = [word_tokenize(text) for text in full_text_lsn]
                pos_tags = [nltk.pos_tag(tt) for tt in text_tokenized]
                focus_words = []
                for i, pos_tag in enumerate(pos_tags):
                    # priortize adjectives and adverbs first
                    fwords = [tag[0] for tag in pos_tag if 'JJ' in tag[1] or 'RB' in tag[1]]
                    if len(fwords) == 0:
                        # if no adjectives or adverbs, then prioritize nouns and verbs
                        fwords = [tag[0] for tag in pos_tag if 'NN' in tag[1] or 'VB' in tag[1] or 'IN' in tag[1]]
                    if len(fwords) == 0:
                        fwords = []
                    
                    # if more than 3 focus words, then randomly select 3
                    if len(fwords) > 3:
                        fwords = random.sample(fwords, 3)

                    focus_words.append(fwords)
                
                # focus on three word window (a phrase) around the focus word instead of one word
                # completely optional (comment out if not needed)
                if len(focus_words) == 0:
                    phrases = []
                else:
                    phrases = []
                    for token, focus_word in zip(text_tokenized, focus_words):
                        if len(focus_word) == 0:
                            continue
                        word = random.sample(focus_word, 1)[0] #focus_word[len(focus_word)//2 - 1] 
                        idx = token.index(word)
                        phrase = token[idx-1:idx+2] if idx > 0 else token[idx:idx+2]
                        phrases.append(phrase)

                focus_words = phrases
            elif self.WEG_type == 'no': # no WEG case
                focus_words = []
            
            print("focus words: ", focus_words)

            
            if self.do_classifier_free_guidance: # modality guidance
                # drop sequence: all_drop, text_drop, audio_drop, spk_drop, apb_drop, lsnid_drop, no_drop
                text_lsn = ['-'*10] * len(text_lsn) + text_lsn + ['-'*10] * len(text_lsn) + ['-'*10] * len(text_lsn) + ['-'*10] * len(text_lsn) + ['-'*10] * len(text_lsn) + text_lsn

                # custom uncond mel for audio drop
                uncond_mel = -90 * torch.ones_like(melspec_lsn)
                uncond_mel[..., 40:45] = 0
                melspec_lsn = torch.cat([uncond_mel, uncond_mel, melspec_lsn, uncond_mel, uncond_mel, uncond_mel, melspec_lsn], dim=0) # audio (bs*2, 128, 80)

                text_spk = ['-'*10] * len(text_spk) + ['-'*10] * len(text_spk) + ['-'*10] * len(text_spk) + text_spk + ['-'*10] * len(text_spk) + ['-'*10] * len(text_spk) + text_spk
                melspec_spk = torch.cat([uncond_mel, uncond_mel, uncond_mel, melspec_spk, uncond_mel, uncond_mel, melspec_spk], dim=0) # audio (bs*2, 128, 80)

                active_passive_bit = torch.cat([2*torch.ones_like(active_passive_bit), # here 2 is used to represent uncond tokens
                                                2*torch.ones_like(active_passive_bit),
                                                2*torch.ones_like(active_passive_bit),
                                                2*torch.ones_like(active_passive_bit),
                                                active_passive_bit,
                                                2*torch.ones_like(active_passive_bit),
                                                active_passive_bit], dim=0)
                
                lsn_id = [0] * len(lsn_id) + [0] * len(lsn_id) + [0] * len(lsn_id) + [0] * len(lsn_id) + [0] * len(lsn_id) + lsn_id + lsn_id 
                
            # breakpoint()
            # aspk, tspk, as_mask, ts_mask, token2word_map_spk, ta_spk = self.text_audio_encoder(text_spk, melspec_spk, person_type='spk-ta')
            aspk, tspk, as_mask, ts_mask, token2word_map_spk,  _ = self.text_audio_encoder(text_spk, melspec_spk, person_type='spk', return_textmap=True)
            alsn, tlsn, al_mask, tl_mask, token2word_map_lsn, _ = self.text_audio_encoder(text_lsn, melspec_lsn, person_type='lsn',  return_textmap=True)

            # breakpoint()

            # WO SEMANTIC CASE:
            # focus_words = []
            # breakpoint()
            text_tokenwordmap = token2word_map_lsn[bs:bs*2]
            if len(focus_words) == 0 or len(focus_words[0]) == 0: # no focus words
                focus_indices = []
            else:
                focus_indices = []
                for b in range(len(text_tokenwordmap)):
                    indices = []
                    for fword in focus_words[b]:
                        indices += [i for i, x in enumerate(text_tokenwordmap[b]) if x == fword]
                    focus_indices.append(indices)
            

            # multiplication factors is num of modalities in modality guidance
            e_lengths = lengths * (self.clf_guidance_drops+1) if self.do_classifier_free_guidance else lengths 
            # if self.vae_type == "no":
            #     motion_spk_emb = motion_spk.permute(2, 0, 1)
            # else:
            #     motion_spk_emb, dist_spk, _ = self.vae.encode(motion_spk, e_lengths)
            #     motion_spk_emb = motion_spk_emb.permute(1, 2, 0, 3) # -> bs, t, bh, dim 

            # motion_spk_emb = motion_spk_emb.permute(1, 2, 0) # 2, bs, 512 => bs, 512, 2

            # when motion is used as spkemb
            # spk_emb = motion_spk_emb

            # when spk text and audio are used as spkemb
            # spk_emb = ta_spk

            # when only spk text is used as spkemb
            spk_emb = tspk
            # breakpoint()

            cond_emb = self.condition_fuser(spk_emb, alsn, tlsn, active_passive_bit, lsn_id)
        elif self.condition == 'textaudio_uncond':
            # REACT CHANGE
            text_lsn = batch["text_lsn"]
            # text_spk = batch["text_spk"]
            # audio = batch["audio"]
            # melspec_spk = batch["melspec_spk"]
            melspec_lsn = batch["melspec_lsn"]
            active_passive_bit = batch["active_passive_lsn"]
            motion_spk = batch["motion_spk"]
            lsn_id = batch["lsn_id"]

            text_cond = ['-'*10] * len(text_lsn) * 2 # uncond_tokens (bs*2, )

            active_passive_cond = (2*torch.ones_like(active_passive_bit)).repeat(2, 1)
            lsn_id = [0] * len(lsn_id) * 2
            uncond_mel = -90 * torch.ones_like(melspec_lsn)
            uncond_mel[..., 40:45] = 0
            melspec_cond = torch.cat([uncond_mel, uncond_mel], dim=0) # audio (bs*2, 128, 80)

            motion_spk_cond = torch.cat([torch.zeros_like(motion_spk), torch.zeros_like(motion_spk)], dim=0)
            # breakpoint()
            # aspk, tspk, as_mask, ts_mask, ta_spk = self.text_audio_encoder(text_cond, melspec_cond, person_type='spk-ta')
            aspk, tspk, as_mask, ts_mask, _ = self.text_audio_encoder(text_cond, melspec_cond, person_type='spk', return_textmap=False)
            alsn, tlsn, al_mask, tl_mask, _ = self.text_audio_encoder(text_cond, melspec_cond, person_type='lsn', return_textmap=False)
            # tspk = tlsn.clone()
            # ts_mask = tl_mask.clone()

            e_lengths = lengths * 2
            
            # if self.vae_type == "no":
            #     motion_spk_emb = motion_spk_cond.permute(2, 0, 1)
            # else:
            #     motion_spk_emb, dist_spk, _ = self.vae.encode(motion_spk_cond, e_lengths)
            #     motion_spk_emb = motion_spk_emb.permute(1, 2, 0, 3) # -> bs, t, bh, dim 
            # motion_spk_emb = motion_spk_emb.permute(1, 2, 0)

            # when motion is used as spkemb
            # spk_emb = motion_spk_emb
            # when spk text and audio are used as spkemb
            # spk_emb = ta_spk
            # when only spk text is used as spkemb
            spk_emb = tspk

            cond_emb = self.condition_fuser(spk_emb, alsn, tlsn, active_passive_cond, lsn_id)
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        
        # diffusion reverse
        with torch.no_grad():
            z, att_mats = self._diffusion_reverse(cond_emb, lengths, cond_masks={'alsn': al_mask, 'tlsn': tl_mask, 'spkemb':ts_mask}, focus_indices=focus_indices)
        # breakpoint()
        with torch.no_grad():
            if self.vae_type == "convofusion":
                ntokens, bs, dim = z.shape
                z = z.reshape(ntokens//2 , 2, bs, dim) # t, bh, bs, dim
                # 
                z = z.permute(1, 2, 0, 3) # bh, bs, t, dim
                
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be convofusion or no")

        # 
        rs_set = {
            "m_rst": feats_rst,
            "m_ref": batch["motion_lsn"].detach(), # feats_ref
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 2, 0, 3), #-> bs, t, bh, dim 
            "test_attention_maps": att_mats,
            "token2word_map_lsn": token2word_map_lsn,
            "token2word_map_spk": token2word_map_spk,
            "focus_words": focus_words 
        }

        # 
        if "motion_lsn" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion_lsn"].detach() 
            with torch.no_grad():
                if self.vae_type == "convofusion":
                    motion_z, dist_m, _ = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm, _ = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref.permute(1, 0, 2)
                    recons_z = feats_rst.permute(1, 0, 2)

            # 
            rs_set["lat_m"] = motion_z.permute(1, 2, 0, 3) # -> bs, t, bh, dim 
            rs_set["lat_rm"] = recons_z.permute(1, 2, 0, 3) # -> bs, t, bh, dim 
            # rs_set["joints_ref"] = joints_ref
        return rs_set


    def allsplit_step(self, split: str, batch, batch_idx):
        # breakpoint()
        
        if split in ["train", "val"]:
            
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                # start = time.time()
                # breakpoint()
                rs_set = self.train_diffusion_forward(batch)
                # print("diffusion time: ", time.time() - start)
                # start = time.time()
                # t2m_rs_set = self.test_diffusion_forward(batch,
                #                                          finetune_decoder=False, split=split)
                # breakpoint()
                # print("gen time: ", time.time() - start)
                # # merge results
                # breakpoint()
                rs_set = {
                    **rs_set,
                    # "gen_m_rst": t2m_rs_set["m_rst"],
                    # "m_ref": t2m_rs_set["m_ref"],
                    # "lat_t": t2m_rs_set["lat_t"],
                    # "lat_m": t2m_rs_set["lat_m"],
                    # "lat_rm": t2m_rs_set["lat_rm"],
                    # "cond_params": self.denoiser.cond_params,
                }
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True, split=split)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    # "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            # print(f"forward time: {time.time() - start:.2f}s")
            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            if split == "test":
                # start = time.time()
                if self.condition in ['text', 'text_uncond']:
                    # use t2m evaluators
                    rs_set = self.t2m_eval(batch)
                elif self.condition == 'action':
                    # use a2m evaluators
                    rs_set = self.a2m_eval(batch)
                # MultiModality evaluation sperately
                elif self.condition == 'text+audio':
                    if self.stage == "vae":
                        rs_set = self.train_vae_forward(batch)
                        rs_set["lat_t"] = rs_set["lat_m"]
                        loss = self.losses[split].update(rs_set)
                    elif self.stage == "diffusion":
                        # rs_set = self.train_diffusion_forward(batch)
                        start = time.time()
                        t2m_rs_set = self.test_diffusion_forward(batch,
                                                                finetune_decoder=False, split=split)
                        print("gen time: ", time.time() - start, "for num of samples: ", len(batch['motion_lsn']))
                        # # merge results
                        rs_set = {
                            # **rs_set,
                            "gen_m_rst": t2m_rs_set["m_rst"],
                            "m_ref": t2m_rs_set["m_ref"],
                            # "lat_t": t2m_rs_set["lat_t"],
                            "lat_m": t2m_rs_set["lat_m"],
                            "lat_rm": t2m_rs_set["lat_rm"],
                            "test_attention_maps": t2m_rs_set["test_attention_maps"],
                            "cond_params": self.denoiser.cond_params,
                            "token2word_map_lsn": t2m_rs_set["token2word_map_lsn"],
                            "token2word_map_spk": t2m_rs_set["token2word_map_spk"],
                            "focus_words": t2m_rs_set["focus_words"]
                        }
                        # loss = self.losses[split].update(rs_set)
                        loss = 0
                        rs_set['m_rst'] = rs_set['gen_m_rst']

                elif self.condition == 'textaudio_uncond' and self.stage == "diffusion":
                    
                    # rs_set = self.train_diffusion_forward(batch)
                    # start = time.time()
                    rs_set = self.test_diffusion_forward(batch,
                                                            finetune_decoder=False, split=split)
                    # print("gen time: ", time.time() - start, "for num of samples: ", len(batch['motion']))
                    loss = 0
                    # # # merge results
                    # rs_set = {
                    #     **rs_set,
                    #     "gen_m_rst": t2m_rs_set["m_rst"],
                    #     "m_ref": t2m_rs_set["m_ref"],
                    #     "lat_t": t2m_rs_set["lat_t"],
                    #     "lat_m": t2m_rs_set["lat_m"],
                    #     "lat_rm": t2m_rs_set["lat_rm"],
                    # }
                    # loss = self.losses[split].update(rs_set)
                    # rs_set['m_rst'] = rs_set['gen_m_rst']
                    

            
            # breakpoint()
            

        # return forward output and loss during test
        if split in ["test"]:
            # breakpoint()
            if self.stage == "vae":
                self.save_npy((rs_set["gt_motion"], rs_set["pred_motion"], batch["length"], batch["name"]))
            else:
                text_lsn = batch["text_lsn"]
                text_spk = batch["text_spk"]
                # audio = batch["audio"]
                audio_spk = batch["audio_spk"]
                audio_lsn = batch["audio_lsn"]
                active_passive_bit = batch["active_passive_lsn"]
                motion_spk = batch["motion_spk"]
                spk_name = batch["spk_name"]
                lsn_name = batch["lsn_name"]
                name = batch["name"]
                melspec_lsn = batch["melspec_lsn"]
                other_mlsn = batch["other_mlsn"]
                full_comb_audio = batch["combined_audio"]
                semantic = batch["sem_lsn"]
                sem_info_lsn = batch["sem_info"]
                # breakpoint()
                if self.condition == "textaudio_uncond":
                    text_lsn = ['-'*10] * len(text_lsn) # uncond_tokens (bs*2, )

                    active_passive_bit = (2*torch.ones_like(active_passive_bit))
                    # lsn_id = [0] * len(lsn_id)
                    uncond_mel = -90 * torch.ones_like(melspec_lsn)
                    uncond_mel[..., 40:45] = 0
                    melspec_lsn = uncond_mel # audio (bs*2, 128, 80)

                    motion_spk = torch.zeros_like(motion_spk)

                self.save_npy((rs_set["m_ref"], 
                            rs_set["m_rst"], 
                            batch["length"], 
                            text_lsn, 
                            text_spk, 
                            audio_lsn, 
                            audio_spk, 
                            active_passive_bit, 
                            motion_spk, 
                            name, 
                            spk_name, 
                            lsn_name, 
                            rs_set["test_attention_maps"],
                            melspec_lsn,
                            other_mlsn,
                            full_comb_audio,
                            dict(lsn=rs_set["token2word_map_lsn"][-len(text_lsn):], spk=rs_set["token2word_map_spk"][-len(text_lsn):]),
                            rs_set["focus_words"],
                            semantic,
                            sem_info_lsn
                        ))
        
        return loss
