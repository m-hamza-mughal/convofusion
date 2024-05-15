import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric




class Losses(Metric):
    """
    MLD Loss
    """

    def __init__(self, vae, mode, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        # self.vae = vae
        self.vae_type = cfg.TRAIN.ABLATION.VAE_TYPE
        self.mode = mode
        self.cfg = cfg
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.stage = cfg.TRAIN.STAGE

        losses = []

        # diffusion loss
        if self.stage in ['diffusion', 'vae_diffusion']:
            # instance noise loss
            losses.append("inst_loss")
            losses.append("x_loss")
            losses.append("condparam_loss")
            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # prior noise loss
                losses.append("prior_loss")

            if self.cfg.LOSS.LAMBDA_GUIDED_ATTENTION != 0.0:
                # guided attention loss
                losses.append("guidedattention_loss")

            if self.cfg.LOSS.LAMBDA_LATENT != 0.0:
                # latent loss
                losses.append("latent_loss")

            losses.append("gen_feature") # TODO: ablation

        if self.stage in ['vae', 'vae_diffusion']:
            # reconstruction loss
            losses.append("recons_feature")
            losses.append('recons_laplace')
            # losses.append("recons_verts")
            # losses.append("recons_joints")
            # losses.append("recons_limb")

            losses.append("gen_feature")
            # losses.append("gen_joints")

            # KL loss
            losses.append("kl_motion")

            # Bone Length Loss
            losses.append("bonelen_feature")

        if self.stage not in ['vae', 'diffusion', 'vae_diffusion']:
            raise ValueError(f"Stage {self.stage} not supported")

        losses.append("total")

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'x':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
            
            elif loss.split('_')[0] == 'condparam':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            
            if loss.split('_')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL

            elif loss.split('_')[0] == 'bonelen':
                if cfg.LOSS.LAMBDA_BL != 0.0:
                    self._losses_func[loss] = BoneLengthLoss(cfg)
                    self._params[loss] = cfg.LOSS.LAMBDA_BL
            
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='none')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='none')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent': # TODO: ablation for latent loss on test outputs
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='none')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            elif loss == 'guidedattention_loss':
                self._losses_func[loss] = GuidedAttentionLoss()
                self._params[loss] = cfg.LOSS.LAMBDA_GUIDED_ATTENTION
            else:
                ValueError("This loss is not recognized.")
            if loss.split('_')[-1] == 'joints':
                self._params[loss] = cfg.LOSS.LAMBDA_JOINT

    def update(self, rs_set):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae", "vae_diffusion"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            # total += self._update_loss("recons_joints", rs_set['joints_rst'],
            #                            rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])
            
            total += self._update_loss("recons_laplace", rs_set['m_laplace_rst'],
                                        rs_set['m_laplace_ref'])
            
            if self.cfg.LOSS.LAMBDA_BL != 0.0:
                total += self._update_loss("bonelen_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])

        if self.stage in ["diffusion", "vae_diffusion"]:
            # predict noise
            if self.predict_epsilon:
                total += self._update_loss("inst_loss", rs_set['noise_pred'],
                                           rs_set['noise'])
            # predict x
            else:
                total += self._update_loss("x_loss", rs_set['pred'],
                                           rs_set['latent'])

            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # loss - prior loss
                # breakpoint()
                total += self._update_loss("prior_loss", rs_set['noise_prior'],
                                           rs_set['dist_m1']) # TODO: FIGURE THIS OUT

            
            if self.cfg.LOSS.LAMBDA_GUIDED_ATTENTION != 0.0:
                # loss - guided attention loss
                total += self._update_loss("guidedattention_loss",
                                           rs_set['train_attention_maps'], None)

            if self.cfg.LOSS.LAMBDA_LATENT != 0.0:
                # loss - latent loss
                total += self._update_loss("latent_loss", rs_set['lat_t'],
                                           rs_set['lat_gt'], weights=rs_set['latloss_weights'])
            # breakpoint()
            # print(rs_set['cond_params'])
            # total += self._update_loss("condparam_loss", rs_set['cond_params'].sum(), torch.tensor(1.0).to(rs_set['cond_params'].device))

        # if self.stage in ["vae_diffusion"]: # TODO: ablation
            # loss
            # noise+text_emb => diff_reverse => latent => decode => motion # this loss is like forcing the gt motion onto the generated one from diffusion ( added lower lambda)
            # breakpoint()
            if self.cfg.LOSS.LAMBDA_GEN != 0.0:
                total += self._update_loss("gen_feature", rs_set['gen_m_rst'],
                                        rs_set['m_ref'])
            # total += self._update_loss("gen_joints", rs_set['gen_joints_rst'],
            #                            rs_set['joints_ref'])

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs, weights=None):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        
        # if loss == "condparam_loss":
        #     print(val.item())
        if loss == "recons_feature":
            # print(loss)
            # val[:, :, list(range(3+ 10*3, 3+ 36*3)) + list(range(3+ 37*3, 3+ 63*3))] *= 10
            val[:, :, :3] *= 10 # scale root loss
            # TODO: add scaling for hands and arm joints
            val[:, :, list(range(5*3, 13*3)) + list(range(23*3, val.shape[-1]))] *= 5 # scale hands and arm joints
            # val *= 5

        if loss == 'recons_laplace': # or loss == 'gen_feature':
            # print(val.mean())
            # breakpoint()
            val[:, :, list(range(5*3, 13*3)) + list(range(23*3, val.shape[-1]))] *= 5
            # val[:, :, list(range(6*3, 9*3)) + list(range(10*3, 13*3)) + list(range(23*3, val.shape[-1]))] *= 5
        
        if loss == 'latent_loss':
            # breakpoint()
            val = weights.reshape(-1, 1, 1) * val
            # print(val.mean())
        # breakpoint()
        val = val.mean()
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        # print(loss, weighted_loss)
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"



class GuidedAttentionLoss(torch.nn.Module):
    def __init__(self, sigma=0.35):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma

    def _make_ga_masks(self, ilens, olens):
        B = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ga_masks[idx, :olen, :ilen] = self._make_ga_mask(
                ilen, olen, self.sigma)
        return ga_masks

    def forward(self, att_ws, args):
        # breakpoint()
        att_ws = att_ws[1:-2] # remove the last two attention maps apb and lsn emb and mspk because they might have global impact
        loss = 0
        for att in att_ws:
            att = torch.mean(att, dim=1) # mean over the layers

            ilens = att.size(2)
            olens = att.size(1)

            ga_masks = self._make_ga_mask(ilens, olens, self.sigma).to(att.device)
            # seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
            losses = att * ga_masks
            # loss = torch.mean(losses.masked_select(seq_masks))
            loss += torch.sum(losses)
        loss = loss / len(att_ws)
        return loss

    @staticmethod
    def _make_ga_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen), indexing='ij')
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen)**2 /
                               (2 * (sigma**2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = sequence_mask(ilens)
        out_masks = sequence_mask(olens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


class BoneLengthLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(BoneLengthLoss, self).__init__()
        dataset_name = cfg.TRAIN.DATASETS[0]
        self.bone_defnition = eval(f"cfg.DATASET.{dataset_name.upper()}.BONES")
        self.n_bones = self.bone_defnition

    def bone_length(self, motion, p1, p2):
        bone_length = ((motion[:, :, p1, :] - motion[:, :, p2, :])**2).sum(-1).sqrt()
        return bone_length

    def forward(self, motion_rst, motion_ref):
        bs, seqlen, nfeats = motion_rst.shape
        rel_motion = motion_rst.reshape(bs, seqlen, nfeats//3, 3)

        loss = torch.tensor(0.0).to(motion_rst.device)
        count = 0
        for j1, j2 in self.bone_defnition:
            if j1 == 0:
                continue
            bone_length_batch = self.bone_length(rel_motion, j1, j2)

            # breakpoint()
            bonelen_var = bone_length_batch.var(dim=1)
            loss += bonelen_var.mean()
            count += 1

        # breakpoint()
        return loss/count




