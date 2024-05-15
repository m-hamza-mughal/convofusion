import os
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import LightningModule
from os.path import join as pjoin
from collections import OrderedDict

import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd


class BaseModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times = []

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if len(self.times) *self.cfg.TEST.BATCH_SIZE % (100) > 0 and len(self.times) > 0:
            print(f"Average time per sample ({self.cfg.TEST.BATCH_SIZE*len(self.times)}): ", np.mean(self.times)/self.cfg.TEST.BATCH_SIZE)
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str, outputs):
        # breakpoint()
        dico = {}

        if split in ["train", "val", "test"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        
        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            # breakpoint()
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        # 
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        # breakpoint()
        # self.save_npy(outputs)
        # self.cfg.TEST.REP_I = self.cfg.TEST.REP_I + 1

        return self.allsplit_epoch_end("test", outputs)

    def on_save_checkpoint(self, checkpoint):
        # don't save clip to checkpoint
        # breakpoint()
        state_dict = checkpoint['state_dict']
        clip_k = []
        for k, v in state_dict.items():
            if 'text_audio_encoder.text_encoder.text_model' in k:
                clip_k.append(k)
        for k in clip_k:
            del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        # restore clip state_dict to checkpoint
        if self.stage != 'vae':
            clip_state_dict = self.text_audio_encoder.text_encoder.text_model.state_dict()
            new_state_dict = OrderedDict()
            for k, v in clip_state_dict.items():
                new_state_dict['text_audio_encoder.text_encoder.text_model.' + k] = v
            for k, v in checkpoint['state_dict'].items():
                if 'text_audio_encoder.text_encoder.text_model' not in k:
                    new_state_dict[k] = v
            checkpoint['state_dict'] = new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # load clip state_dict to checkpoint
        if self.stage != 'vae':
            clip_state_dict = self.text_audio_encoder.text_encoder.text_model.state_dict()
            new_state_dict = OrderedDict()
            for k, v in clip_state_dict.items():
                # new_state_dict['text_encoder.' + k] = v
                new_state_dict['text_audio_encoder.text_encoder.text_model.' + k] = v
            for k, v in state_dict.items():
                if 'text_audio_encoder.text_encoder.text_model' not in k:
                    new_state_dict[k] = v
                new_state_dict[k] = v
                
        else:
            new_state_dict = state_dict
            
        # breakpoint()
        super().load_state_dict(new_state_dict, strict)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    def save_npy(self, outputs):
        
        cfg = self.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.model_type),
                str(cfg.NAME),
                "samples_" + cfg.TIME,
            ))
        if cfg.TEST.SAVE_PREDICTIONS:
            if cfg.TEST.DATASETS[0].lower() in ["beatdnd"]:
                gt_full = outputs[0].cpu().numpy() # m_ref
                pred_full = outputs[1].cpu().numpy() # m_rst
                lengths = outputs[2] # length
                if self.stage == 'vae':
                    keyids = outputs[3] # keyids
                if self.stage != 'vae':
                    texts_lsn = outputs[3] # text_lsn
                    texts_spk = outputs[4] # text_spk
                    audios_lsn = outputs[5].cpu().numpy() # audio_lsn
                    audios_spk = outputs[6].cpu().numpy() # audio_spk
                    active_passive_bit = outputs[7] # active_passive_bit
                    motions_spk = outputs[8].cpu().numpy() # motion_spk
                    keyids = outputs[9] # keyids
                    spk_name = outputs[10] # spk_name
                    lsn_name = outputs[11] # lsn_name
                    att_maps =outputs[12] # test att maps
                    melspec_lsn = outputs[13].cpu().numpy() # melspec_lsn
                    # 
                    token2word_map = outputs[16]
                    focus_words = outputs[17]
                    sem_lsn = outputs[18]
                    sem_lsn = sem_lsn.cpu().numpy() if sem_lsn is not None else None
                    sem_info_lsn = outputs[19] if len(outputs) > 19 else None

                    
                    
                    # breakpoint()
                    att_names = ["att_spk", "att_alsn", "att_tlsn", "att_apb", "att_lsnemb"]
                    assert len(token2word_map['spk']) == len(texts_spk)
                for i in range(len(gt_full)):
                    name = f"{keyids[i]}"

                    sample_dir = os.path.join(output_dir, name)
                    os.makedirs(name=sample_dir, exist_ok=True)

                    gt = gt_full[i][:lengths[i]]
                    pred = pred_full[i][:lengths[i]]
                    # breakpoint()
                    if self.stage != 'vae':
                        motion_spk = motions_spk[i][:lengths[i]]

                    # reverse normalization and reprsentation
                    # if self.trainer.datamodule.test_dataset.motion_rep == "6D":
                    #     pred_trans = pred[:, :3] #* 100 # root translation -> m to cm
                    #     gt_trans = gt[:, :3] #* 100 # root translation -> m to cm

                    #     # breakpoint()
                    #     # gt_torch = torch.from_numpy(gt)
                    #     # pred_torch = torch.from_numpy(pred)
                    #     # gt_joints = self.trainer.datamodule.rep6d2joints(gt_torch)
                    #     # pred_joints = self.trainer.datamodule.rep6d2joints(pred_torch)

                    #     pred_rot = self.trainer.datamodule.rep6d2euler(pred[:, 3:])
                    #     gt_rot = self.trainer.datamodule.rep6d2euler(gt[:, 3:])

                    #     pred = np.concatenate((pred_trans, pred_rot), axis=1)
                    #     gt = np.concatenate((gt_trans, gt_rot), axis=1)
                    # elif self.trainer.datamodule.test_dataset.motion_rep == "euler":
                    #     pred[:, 3:] = pred[:, 3:] * 180 # root rotation (euler angle) -> [-1, 1]
                    #     pred[:, :3] = pred[:, :3] * 100 # root translation -> cm to m

                    #     gt[:, 3:] = gt[:, 3:] * 180 # root rotation (euler angle) -> [-1, 1]
                    #     gt[:, :3] = gt[:, :3] * 100 # root translation -> cm to m
                    assert self.trainer.datamodule.test_dataset.motion_rep == "pos"
                    p_keypoints3d = pred.reshape(-1, self.njoints, 3)
                    p_keypoints3d = p_keypoints3d / 3
                    # p_keypoints3d[:, 23:, :] = p_keypoints3d[:, 23:, :] / 10
                    p_keypoints3d[:, 43:, :] = p_keypoints3d[:, 43:, :] + p_keypoints3d[:, [11], :]  
                    p_keypoints3d[:, 23:43, :] = p_keypoints3d[:, 23:43, :] + p_keypoints3d[:, [7], :]  # 7 is wrist instead of 8
                    p_keypoints3d[:, 1:, :] = p_keypoints3d[:, 1:, :] + p_keypoints3d[:, :1, :] 

                    g_keypoints3d = gt.reshape(-1, self.njoints, 3)
                    g_keypoints3d = g_keypoints3d / 3
                    # g_keypoints3d[:, 23:, :] = g_keypoints3d[:, 23:, :] / 10
                    g_keypoints3d[:, 43:, :] = g_keypoints3d[:, 43:, :] + g_keypoints3d[:, [11], :] # 11 is wrist instead of 12
                    g_keypoints3d[:, 23:43, :] = g_keypoints3d[:, 23:43, :] + g_keypoints3d[:, [7], :]  # 7 is wrist instead of 8
                    g_keypoints3d[:, 1:, :] = g_keypoints3d[:, 1:, :] + g_keypoints3d[:, :1, :]

                    pred = p_keypoints3d
                    gt = g_keypoints3d
                    if self.stage != 'vae':
                        spk_keypoints3d = motion_spk.reshape(-1, self.njoints, 3)
                        spk_keypoints3d = spk_keypoints3d / 3
                        # spk_keypoints3d[:, 23:, :] = spk_keypoints3d[:, 23:, :] / 10
                        spk_keypoints3d[:, 43:, :] = spk_keypoints3d[:, 43:, :] + spk_keypoints3d[:, [11], :]
                        spk_keypoints3d[:, 23:43, :] = spk_keypoints3d[:, 23:43, :] + spk_keypoints3d[:, [7], :]
                        spk_keypoints3d[:, 1:, :] = spk_keypoints3d[:, 1:, :] + spk_keypoints3d[:, :1, :]
                        motion_spk = spk_keypoints3d

                    # save ground truth results
                    npypath = os.path.join(sample_dir, 'gt.npy')
                    np.save(npypath, gt)
                    # save ground truth results as bvh (here if needed)

                    # save predictions results
                    npypath = os.path.join(sample_dir, 'pred.npy')
                    np.save(npypath, pred)
                    # save predictions results as bvh (here if needed)

                    
                    # save conditions
                    if self.stage != 'vae':
                        # breakpoint()
                        if isinstance(att_maps, dict):

                            lsn_wordmap = ",".join(token2word_map['lsn'][0])
                            spk_wordmap = ",".join(token2word_map['spk'][0])
                            with open(os.path.join(sample_dir, 'lsn_wordmap.txt'), 'w') as f:
                                f.write(lsn_wordmap)
                            with open(os.path.join(sample_dir, 'spk_wordmap.txt'), 'w') as f:
                                f.write(spk_wordmap)
                            
                            for t, att_map in att_maps.items():
                                for idx, name in enumerate(att_names):
                                    # print(t, name, idx, len(att_map[idx]))
                                    am = att_map[idx][0].cpu().numpy()
                                    att_dir = os.path.join(sample_dir, name)
                                    os.makedirs(att_dir, exist_ok=True)
                                    npypath = os.path.join(att_dir, f'att_{t}.npy')
                                    np.save(npypath, am)




                        if att_maps is not None and not isinstance(att_maps, dict):
                            for idx, name in enumerate(att_names):
                                # breakpoint()
                                # if len(att_maps[idx]) < 32:
                                    # breakpoint()
                                att_map = att_maps[idx][i].cpu().numpy()

                                # save matplotlib imshow subplot of att_map which is a list of 2d arrays
                                fig = plt.figure(figsize=(len(att_map)*8, 10 ))
                                for j in range(len(att_map)):
                                    ax = fig.add_subplot(1, len(att_map), j+1)
                                    
                                    if name == 'att_spk':
                                        ax_j = ax.imshow(att_map[j], aspect=0.5)
                                        
                                        ax.set_xticks(np.arange(len(token2word_map['spk'][i])))
                                        ax.set_xticklabels(labels = token2word_map['spk'][i], rotation=90, fontsize=7)
                                    elif name == 'att_tlsn':
                                        # breakpoint()
                                        ax_j = ax.imshow(att_map[j], aspect=0.5)
                                        ax.set_xticks(np.arange(len(token2word_map['lsn'][i])))
                                        ax.set_xticklabels(labels = token2word_map['lsn'][i], rotation=90, fontsize=7)
                                    else:
                                        ax_j = ax.imshow(att_map[j], aspect='auto')
                                    
                                    plt.colorbar(ax_j, shrink=0.5)
                                    plt.title('layer' + str(j))
                                
                                plt.savefig(os.path.join(sample_dir, f'{name}.png'))
                                
                                plt.close()

                        if len(focus_words) > 0:
                            focus_words_lsn = focus_words[i]
                            # 
                            focus_words_lsn = [','.join(ts) for ts in focus_words_lsn]
                            focus_words_lsn = "\n".join(focus_words_lsn)


                            # focus_words_lsn = ", ".join(focus_words_lsn)
                            with open(os.path.join(sample_dir, 'focus_words_lsn.txt'), 'w') as f:
                                f.write(focus_words_lsn)

                        # save motion_spk results
                        npypath = os.path.join(sample_dir, 'spk_motion.npy')
                        np.save(npypath, motion_spk)
                        # save motion_spk results as bvh (here if needed)

                        wavpath = os.path.join(sample_dir, 'lsn_audio.wav')
                        sf.write(wavpath, audios_lsn[i], samplerate=16000)

                        wavpath = os.path.join(sample_dir, 'spk_audio.wav')
                        sf.write(wavpath, audios_spk[i], samplerate=16000)

                        # combine both audio and save
                        wavpath = os.path.join(sample_dir, 'combined_audio.wav')
                        sf.write(wavpath, audios_lsn[i] + audios_spk[i], samplerate=16000)

                        if sem_lsn is not None:
                            np.save(os.path.join(sample_dir, 'sem_lsn.npy'), sem_lsn[i])


                        # breakpoint()
                        if sem_info_lsn is not None:
                            sem_info_lsn_sample = sem_info_lsn[i]
                            sem_info_df = pd.DataFrame(sem_info_lsn_sample)
                            sem_info_df.to_csv(os.path.join(sample_dir, 'sem_info_lsn.csv'), index=False, sep='\t')

                        # save text
                        textpath = os.path.join(sample_dir, 'lsn_text.txt')
                        with open(textpath, 'w') as f:
                            f.write(texts_lsn[i])

                        textpath = os.path.join(sample_dir, 'spk_text.txt')
                        with open(textpath, 'w') as f:
                            f.write(texts_spk[i])
                        
                        
                        # save melspec as image
                        plt.figure(figsize=(10, 4))
                        # breakpoint()
                        plt.imshow(melspec_lsn[i].T[::-1], vmin=-90, vmax=0)
                        plt.colorbar()
                        plt.savefig(os.path.join(sample_dir, 'lsn_melspec.png'))
                        plt.close()
                        
                        # breakpoint()
                        # save meta
                        metapath = os.path.join(sample_dir, 'meta.txt')
                        with open(metapath, 'w') as f:
                            f.write(f"lsn: {lsn_name[i]}\nspk: {spk_name[i]}\nactive_passive_bit: {active_passive_bit[i]}")
                        
                    # if i > 2:
                    #     break

            
