import inspect
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from tqdm import tqdm

from convofusion.callback import ProgressLogger
from convofusion.config import parse_args
from convofusion.data.get_data import get_datasets
from convofusion.models.get_model import get_model
from convofusion.utils.logger import create_logger
from torch.utils.data import DataLoader

from convofusion.models.operator import GaussianSmoothing
from convofusion.models.tools import weg
from nltk.tokenize import word_tokenize
import nltk
import random

# torch.manual_seed(0)
    
def diffusion_reverse_forecast(model, encoder_hidden_states, lengths=None, preseq=None, cond_masks=dict(), focus_indices=[]):
        # preseq shape is (bsz, seq_len, feat_dim)
        # init latents
        # breakpoint()
        bsz = encoder_hidden_states[0].shape[0]
        # breakpoint()
        if model.do_classifier_free_guidance:
            guidance_bs_mulitplier = model.clf_guidance_drops + 1
            bsz = bsz // guidance_bs_mulitplier
        
        # DEBUG: check this random tensor initialization - done
        # random_tensor = np.random.randn(bsz, 16, model.latent_dim[-1])
        init_noise = torch.randn(
                (bsz, 16, model.latent_dim[-1]),
                # random_tensor, # HARD CODED - changed from 16 for ikhsi's idea
                device=encoder_hidden_states[0].device,
                dtype=torch.float,
                # generator = torch.manual_seed
            )

        # scale the initial noise by the standard deviation required by the scheduler
        init_noise = init_noise * model.scheduler.init_noise_sigma

        if preseq is not None:
            preseq_len = preseq.shape[1]
            assert preseq.shape[0] == init_noise.shape[0]

        # set timesteps
        model.scheduler.set_timesteps(
            model.cfg.model.scheduler.num_inference_timesteps)
        timesteps = model.scheduler.timesteps.to(encoder_hidden_states[0].device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(model.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = model.cfg.model.scheduler.eta

        # reverse
        latents = init_noise
        attention_matrices = dict()
        for i, t in enumerate(timesteps):

            if preseq is not None:
                # breakpoint()
                preseq_prenoise = preseq.clone()
                preseq_noise = init_noise.clone()

                noised_preseq = model.noise_scheduler.add_noise(preseq_prenoise, preseq_noise[:, :preseq_len, :], t)
                latents[:, :preseq_len, :] = noised_preseq

            if len(focus_indices) > 0:
                with torch.inference_mode(False):
                    # TODO: move to config
                    # breakpoint()
                    scale_factor = 100
                    scale_range = (1., 0.5)
                    max_iter_to_alter = 800
                    thresholds = {0: 0.05, 200: 0.4, 400: 0.6, 600:0.8}
                    max_refinement_steps = 300

                    scale_range = np.linspace(scale_range[0], scale_range[1], len(model.scheduler.timesteps))

                    # 
                    latents = latents.clone().detach().requires_grad_(True)

                    text_only_encoder_hidden_states = [ enc.chunk(guidance_bs_mulitplier)[1] for enc in encoder_hidden_states]
                    text_only_cond_masks = {k: v.chunk(guidance_bs_mulitplier)[1] if v is not None else v for k, v in cond_masks.items()}
                    # breakpoint()
                    noise_pred_text, text_only_att_mats = model.denoiser(
                        sample=latents,
                        timestep=t,
                        encoder_hidden_states=text_only_encoder_hidden_states,
                        lengths=lengths,
                        mem_mask_dict=text_only_cond_masks
                    ) #[0]
                    model.denoiser.zero_grad()
                    # breakpoint()
                    eot_indices = torch.argmax(text_only_cond_masks['tlsn'].int(),  dim=1) - 1

                    # text_only_att_mats = [att_mat.chunk(guidance_bs_mulitplier)[1] for att_mat in att_mats]
                    # check the shapes of the attention matrices
                    text_att_mats = text_only_att_mats[2]

                    # breakpoint()

                    # aggregate and Get max activation value for each focus token defined by focus indices
                    text_att_mats = weg.aggregate_attentions(text_att_mats)
                    max_attention_at_indices = weg.get_max_attention_at_indices(text_att_mats, focus_indices, smooth_attentions=True, normalize_eot=True, eot_indices=eot_indices)
                    
                    loss, losses = weg.compute_attention_focus_loss(max_attention_at_indices)

                    # TODO: add iterative refinement step here after testing
                    if i in thresholds.keys() and loss > 1. - thresholds[i]:
                        del noise_pred_text
                        torch.cuda.empty_cache()
                        loss, latents, max_attention_at_indices = model.iterative_refinement_step(latents=latents,
                                        indices_to_alter=focus_indices,
                                        loss=loss,
                                        threshold=thresholds[i],
                                        encoder_hidden_states=text_only_encoder_hidden_states,
                                        lengths=lengths,
                                        cond_masks=text_only_cond_masks,
                                        step_size=scale_factor * np.sqrt(scale_range[i]),
                                        t=t,
                                        max_refinement_steps=max_refinement_steps,
                                        normalize_eot=True, eot_indices=eot_indices)
                        
                    
                    if i < max_iter_to_alter:
                        # compute loss w.r.t latent
                        loss, losses = weg.compute_attention_focus_loss(max_attention_at_indices)
                        if loss.all() != 0:
                            # breakpoint()
                            latents = weg.update_latent(latents=latents, loss=loss, lr=scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss:0.4f}')
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents
            latent_model_input = (torch.cat(
                [latents] *
                guidance_bs_mulitplier) if model.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * guidance_bs_mulitplier if model.do_classifier_free_guidance else lengths)
            # 
            noise_pred, att_mats = model.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
                mem_mask_dict=cond_masks
            ) #[0]

            att_mats = [att_mat.chunk(guidance_bs_mulitplier)[-1] for att_mat in att_mats] 
            
            attention_matrices[t.item()] = att_mats

            # perform modality guidance
            if model.do_classifier_free_guidance:
                # drop sequence: all_drop, text_drop, audio_drop, spk_drop, apb_drop, lsnid_drop, no_drop
                noise_pred_uncond, noise_pred_textonly, noise_pred_audioonly, noise_pred_spkonly, \
                    noise_pred_apbonly, noise_pred_lsnidonly, noise_pred_fullcond = noise_pred.chunk(guidance_bs_mulitplier)
                noise_pred_text = model.guidance_scale * 1 * (noise_pred_textonly - noise_pred_uncond)
                noise_pred_audio = model.guidance_scale * 1 * (noise_pred_audioonly - noise_pred_uncond)
                noise_pred_spk = model.guidance_scale * 1 * (noise_pred_spkonly - noise_pred_uncond)
                noise_pred_apb = model.guidance_scale  * 1 * (noise_pred_apbonly - noise_pred_uncond)
                noise_pred_lsnid = model.guidance_scale  * 1 * (noise_pred_lsnidonly - noise_pred_uncond)
                noise_pred_all = model.guidance_scale * 0 * (noise_pred_fullcond - noise_pred_uncond) # multiply by 0.1?
                
                # 
                noise_pred =  noise_pred_uncond + (noise_pred_text + noise_pred_audio + noise_pred_spk + noise_pred_apb + noise_pred_lsnid + noise_pred_all)
            

            # breakpoint()
            # att_mats = [att_mat.chunk(2)[1] for att_mat in att_mats]
            latents = model.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample   # input latents is x_t and output is x_{t-1}

        # [batch_size, 16, latent_dim] -> [16, batch_size, latent_dim]
        # latents = latents[:, :16, :] # HARD CODED - changed for ikhsi's idea
        latents = latents.permute(1, 0, 2)
        return latents, att_mats

def process_text(seg_lsn_batch, chunk_tstart, chunk_tend):
    # breakpoint()
    text_batch = []
    chunk_len = chunk_tend - chunk_tstart
    chunk_mid = (chunk_tstart + chunk_tend) / 2
    for seg_lsn in seg_lsn_batch:
        if seg_lsn == '-'*10:
            text_batch.append(seg_lsn)
            continue
        
        chunk_text = []
        # 
        for s_idx, seg in enumerate(seg_lsn):
            seg_start, seg_end = float(seg[0][0]), float(seg[0][1])
            # print(seg_start, seg_end, chunk_tstart, chunk_tend, seg[1], seg_start <= chunk_tstart, seg_start >= (chunk_tstart - 1), seg_end >= chunk_mid, seg_end <= chunk_tend)
            if seg_start >= chunk_tstart and seg_end <= chunk_tend:
                # 
                chunk_text.append(seg[1])
            elif seg_end >= chunk_mid and seg_end <= chunk_tend and ((seg_start < (chunk_tstart - chunk_len/2) and s_idx > 0) or (seg_start < chunk_tstart and s_idx == 0)):
                # 
                chunk_text.append(seg[1])
            
            elif seg_start >= (chunk_tstart-1) and seg_start < chunk_tstart and seg_end <= (chunk_tend+1) and seg_end > chunk_tend:
                # 
                chunk_text.append(seg[1])
                # break
            elif seg_start >= chunk_tstart and seg_start <= chunk_mid and seg_end <= (chunk_tend + 1) and seg_end >= chunk_tend:
                # 
                chunk_text.append(seg[1])
                # break
            elif seg_start <= chunk_tstart and seg_start >= (chunk_tstart - 1) and seg_end >= chunk_mid and seg_end <= chunk_tend:
                # 
                chunk_text.append(seg[1])
                # break
            elif seg_start > chunk_mid and seg_start <= (chunk_tend-1) and seg_end <= (chunk_tend+1):
                # 
                chunk_text.append(seg[1])
                # break
            elif seg_start >= (chunk_tstart-1) and seg_end >= (chunk_tstart+2) and seg_end < chunk_mid:
                # 
                chunk_text.append(seg[1])
                # break

            # if len(chunk_text) > 0:
            #     print(seg_start, seg_end, chunk_tstart, chunk_tend,  ' '.join(chunk_text))
            

        # print(chunk_text, '\n++++')
        sample_text = ' '.join(chunk_text)
        
        # print(sample_text, '\n++++')
        text_batch.append(sample_text)
    return text_batch


def process_samples(batch, model, cfg, logger, output_dir):
    # breakpoint()
    # print(batch.keys())
    
    l_lengths = batch["length"]
    l_feats_ref = batch["motion_lsn"]
    
    l_text_lsn = batch["text_lsn"].copy()
    l_textspk_batch = batch["text_spk"].copy()

    l_melspec_spk = batch['melspec_spk'].clone().to(model.device)
    l_melspec_lsn = batch["melspec_lsn"].clone().to(model.device)

    l_active_passive_bit = batch["active_passive_lsn"].clone().to(model.device)
    l_motion_spk = batch["motion_spk"].clone().to(model.device)
    l_lsn_id = batch["lsn_id"] #.to(model.device)
    l_audio_lsn = batch["audio_lsn"].to(model.device)

    l_seg_lsn = batch["seg_lsn"]
    l_seg_spk = batch["seg_spk"]
    name_batch = batch["name"]
    
    l_audiospk_batch = batch["audio_spk"]
    l_comb_audio = batch["combined_audio"]
    spk_names = batch["spk_name"]
    lsn_names = batch["lsn_name"]
    
    # breakpoint()
    motion_len = 128
    time_len = motion_len / 25
    
    n_parts = l_feats_ref.shape[1] // motion_len
    n_iters = 2 * n_parts - 1
    # print('n_iters: ', n_iters, 'n_parts: ', n_parts)

    melspec_len = l_melspec_lsn.shape[1] // n_parts
    apb_len = l_active_passive_bit.shape[1] // n_parts
    motion_spk_len = motion_len
    audio_len = l_audio_lsn.shape[1] // n_parts
    

    preseq = None
    prev = None
    # breakpoint()
    feat_rst_list = []

    for chunk_idx in tqdm(range(n_iters)):

        chunk_tstart, chunk_tend = (chunk_idx/2)*time_len, ((chunk_idx/2)+1)*time_len
        # 
        # 
        text_lsn = process_text(l_seg_lsn, chunk_tstart, chunk_tend)
        # print(chunk_text)
        feats_ref = l_feats_ref[:, int((chunk_idx/2)*motion_len):int(((chunk_idx/2)+1)*motion_len), :].to(model.device)
        # breakpoint()
        lengths = [motion_len] * len(l_lengths)
        # chunk_ref = chunk_feats
        # chunk_text = text_lsn # TEMPORARY
        melspec_lsn = l_melspec_lsn[:, int((chunk_idx/2)*melspec_len):int(((chunk_idx/2)+1)*melspec_len) + 1, :].to(model.device)

        melspec_spk = l_melspec_spk[:, int((chunk_idx/2)*melspec_len):int(((chunk_idx/2)+1)*melspec_len) + 1, :].to(model.device)
        text_spk = process_text(l_seg_spk, chunk_tstart, chunk_tend)
        # chunk_textspk = textspk_batch # TEMPORARY

        active_passive_bit = l_active_passive_bit[:, int((chunk_idx/2)*apb_len):int(((chunk_idx/2)+1)*apb_len)].to(model.device)
        motion_spk = l_motion_spk[:, int((chunk_idx/2)*motion_spk_len):int(((chunk_idx/2)+1)*motion_spk_len), :].to(model.device)
        lsn_id = l_lsn_id
        audio_lsn = l_audio_lsn[:, int((chunk_idx/2)*audio_len):int(((chunk_idx/2)+1)*audio_len)].to(model.device)

        full_text_lsn = text_lsn.copy()
        bs = len(text_lsn)

        # breakpoint() 
        # TODO check if there are repititions of focus words along chunks and how do they react with inpainting
        if chunk_idx != 0:
            full_text_lsn = process_text(l_seg_lsn, ((chunk_idx + 1)/2)*time_len, ((chunk_idx/2)+1)*time_len)

        
        if model.WEG_type == 'semantic':
            # utilize semantic information to select focus words from BEAT dataset
            assert model.datamodule._sample_set.dataset_select == 'beat', "Semantic WEG only supported for BEAT dataset"
            focus_words = [[entry['word'] for entry in batch['sem_info'][i] if isinstance(entry['word'], str)] for i in range(bs)]
        elif model.WEG_type == 'random':
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
        elif model.WEG_type == 'no': # no WEG case
            focus_words = []
            
        # breakpoint()
        # TODO check all dims are correct
        # TODO check if there are repititions of focus words along chunks and how do they react with inpainting

        if model.condition == 'text+audio':
            if model.do_classifier_free_guidance:
                # breakpoint()
                # text_cond = [['-'*10] * len(texts) + texts for texts in [text_spk, text_lsn]] # uncond_tokens + text (bs*2, )
                
                # drop sequence: all_drop, text_drop, audio_drop, spk_drop, apb_drop, lsnid_drop, no_drop
                text_lsn = ['-'*10] * len(text_lsn) + text_lsn + ['-'*10] * len(text_lsn) + ['-'*10] * len(text_lsn) + ['-'*10] * len(text_lsn) + ['-'*10] * len(text_lsn) + text_lsn

                uncond_mel = -90 * torch.ones_like(melspec_lsn)
                uncond_mel[..., 40:45] = 0
                melspec_lsn = torch.cat([uncond_mel, uncond_mel, melspec_lsn, uncond_mel, uncond_mel, uncond_mel, melspec_lsn], dim=0) # audio (bs*2, 128, 80)

                text_spk = ['-'*10] * len(text_spk) + ['-'*10] * len(text_spk) + ['-'*10] * len(text_spk) + text_spk + ['-'*10] * len(text_spk) + ['-'*10] * len(text_spk) + text_spk
                melspec_spk = torch.cat([uncond_mel, uncond_mel, uncond_mel, melspec_spk, uncond_mel, uncond_mel, melspec_spk], dim=0) # audio (bs*2, 128, 80)

                
                # motion_spk = torch.cat([torch.zeros_like(motion_spk), motion_spk], dim=0)
                # breakpoint()

                active_passive_bit = torch.cat([2*torch.ones_like(active_passive_bit), 
                                                2*torch.ones_like(active_passive_bit),
                                                2*torch.ones_like(active_passive_bit),
                                                2*torch.ones_like(active_passive_bit),
                                                active_passive_bit,
                                                2*torch.ones_like(active_passive_bit),
                                                active_passive_bit], dim=0)
                
                lsn_id = [0] * len(lsn_id) + [0] * len(lsn_id) + [0] * len(lsn_id) + [0] * len(lsn_id) + [0] * len(lsn_id) + lsn_id + lsn_id 
            
            aspk, tspk, as_mask, ts_mask, token2word_map_spk,  _ = model.text_audio_encoder(text_spk, melspec_spk, person_type='spk', return_textmap=True)
            alsn, tlsn, al_mask, tl_mask, token2word_map_lsn, _ = model.text_audio_encoder(text_lsn, melspec_lsn, person_type='lsn',  return_textmap=True)

            

            text_tokenwordmap = token2word_map_lsn[bs:bs*2]
            if len(focus_words) == 0: 
                focus_indices = []
            else:
                focus_indices = []
                for b in range(len(text_tokenwordmap)):
                    indices = []
                    for fword in focus_words[b]:
                        indices += [i for i, x in enumerate(text_tokenwordmap[b]) if x == fword]
                    focus_indices.append(indices)

            e_lengths = lengths * (model.clf_guidance_drops+1) if model.do_classifier_free_guidance else lengths

            assert model.vae_type != "no", "VAE is always used"
            # motion_spk_emb, _, _ = model.vae.encode(chunk_motion_spk, e_lengths)
            # motion_spk_emb = motion_spk_emb.permute(1, 2, 0, 3) # -> bs, t, bh, dim 
            # motion_spk_emb = motion_spk_emb.permute(1, 2, 0) # 2, bs, 512 => bs, 512, 2

            # DEBUG: add classifier free guidance here
            
            # when motion is used as spkemb
            # spk_emb = motion_spk_emb
            # when spk text and audio are used as spkemb
            # spk_emb = ta_spk
            # when only spk text is used as spkemb
            spk_emb = tspk
            # breakpoint()
            cond_emb = model.condition_fuser(spk_emb, alsn, tlsn, active_passive_bit, lsn_id)
        elif model.condition == 'textaudio_uncond':
            pass
        else:
            raise TypeError(f"condition type {model.condition} not supported")
        # breakpoint()
        # diffusion reverse

        with torch.no_grad():
            # breakpoint()
            z, att_mats = diffusion_reverse_forecast(model, cond_emb, lengths, preseq, cond_masks={'alsn': al_mask, 'tlsn': tl_mask, 'spkemb':ts_mask}, focus_indices=focus_indices)
            print(z.shape)

        # breakpoint()
        preseq = z.clone()
        preseq = preseq[preseq.shape[0]//2:, :, :]
        preseq = preseq.permute(1, 0, 2)

        with torch.no_grad():
            ntokens, bs, dim = z.shape
            z = z.reshape(ntokens//2 , 2, bs, dim) # t, bh, bs, dim
            # z = z.permute(2, 1, 0)
            z = z.permute(1, 2, 0, 3) # bh, bs, t, dim
            
            # motion_z_orig = z.clone()
            # t_mse = []
            # for t in range(motion_z_orig.shape[-2]):
            #     # breakpoint()
            #     current_t = motion_z_orig[:, :, [t]*(motion_z_orig.shape[-2]-1), :]
            #     all_t = motion_z_orig[:, :, [i for i in range(motion_z_orig.shape[-2]) if i != t], :]
            #     t_mse.append(torch.mean((current_t - all_t)**2))
            # print('mse between timestamps: ', torch.mean(torch.stack(t_mse)).item())
            feats_rst = model.vae.decode(z, lengths)
            if prev is not None:
                # breakpoint()
                feats_rst[:, :, :3] = feats_rst[:, :, :3] - feats_rst[:, :1, :3] * torch.tensor([1, 0, 1]).to(feats_rst.device)
                # TODO: check if you can make root of feats_rst ego-centric
                feats_rst[:, :, :3] = feats_rst[:, :, :3] + (prev[:, :1, :3] * torch.tensor([1, 0, 1]).to(prev.device))

        feat_rst_list.append(feats_rst.detach().cpu())
        prev = feats_rst[:, motion_len//2:, :]


        names = [n + '+' + str(chunk_idx) for n in name_batch]
        
        # model.save_npy((chunk_feats, feats_rst, chunk_lengths, names))
        # breakpoint()
        if model.do_classifier_free_guidance:
            
            melspec_spk = melspec_spk.chunk(model.clf_guidance_drops+1)[-1]  # audio (bs*2, 128, 80)
            melspec_lsn = melspec_lsn.chunk(model.clf_guidance_drops+1)[-1] # audio (bs*2, 128, 80)

            active_passive_bit = active_passive_bit.chunk(model.clf_guidance_drops+1)[-1]
            text_spk = text_spk[-len(melspec_spk):]
            text_lsn = text_lsn[-len(melspec_spk):]

            lsn_id = lsn_id[-len(melspec_spk):]


        text_full_spk = l_textspk_batch
        
        audio_spk = l_audiospk_batch[:, int((chunk_idx/2)*audio_len):int(((chunk_idx/2)+1)*audio_len)]
        
        comb_audio = l_comb_audio[:, int((chunk_idx/2)*audio_len):int(((chunk_idx/2)+1)*audio_len)]

        model.save_npy((feats_ref, 
                            feats_rst, 
                            lengths, 
                            text_lsn, 
                            text_full_spk, 
                            audio_lsn, 
                            audio_spk, 
                            active_passive_bit, 
                            motion_spk, 
                            names, 
                            spk_names, 
                            lsn_names, 
                            att_mats,
                            melspec_lsn,
                            None,
                            l_comb_audio,
                            dict(lsn=token2word_map_lsn[-len(text_lsn):], spk=token2word_map_spk[-len(text_lsn):]),
                            focus_words,
                            None # sem_lsn None
                        ))

    # # breakpoint()

        



def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    # create logger
    logger = create_logger(cfg, phase="test")
    
    cfg.NAME = 'test_diffrollout_' + cfg.NAME
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        #     str(x) for x in cfg.DEVICE)
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    cfg.DATASET.SAMPLER.MAX_LEN = 128 * 6
    cfg.DATASET.SAMPLER.MIN_LEN = 128 * 6
    # breakpoint()
    cfg.model.audio_encoder.params.max_seq_len = 128
    cfg.DATASET.BEATDND.ROOT = ['./datasets/beat_english_v0.2.1/','./datasets/utterance_dataset_30sec']
    cfg.DATASET.BEATDND.SPLIT_ROOT = ['./datasets/beat_english_v0.2.1/','./datasets/utterance_dataset_30sec']

    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets)
    model.cuda()
    logger.info("model {} loaded".format(cfg.model.model_type))

    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cuda")["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    bs = cfg.TEST.BATCH_SIZE
    
    dataloader = datasets.test_dataloader()

    for batch in dataloader:
        # breakpoint()
        process_samples(batch, model, cfg, logger, output_dir)
        # process_samples(batch, model, cfg, logger, output_dir, wospk_ablation=True)


if __name__ == "__main__":
    main()



