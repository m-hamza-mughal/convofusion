import torch
import torch.nn as nn
import torch.nn.functional as F

from convofusion.models.operator.position_encoding import PositionEmbeddingSine1D


class TextAudioMotionFuser(nn.Module):
    def __init__(self, cfg, out_dim):
        super(TextAudioMotionFuser, self).__init__()
        lat0, lat1 = cfg.model.latent_dim
        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")
        
        self.out_dim = out_dim
        
        self.active_passive_emb = nn.Embedding(3, out_dim)
        self.lsn_id_emb = nn.Embedding(5 + 1 + 30, out_dim) # 5 lsn + 1 unconditional
        
        self.latent_proj = nn.Sequential(
            nn.Linear(lat1 if self.vae_type != "no" else 189, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.GELU(),
        )
        
        

    def forward(self, spkemb, alsn, tlsn, active_passive_bit, lsn_id):
        """
        mm_emb_spk: (batch_size, seq_len, dim)
        mm_emb_lsn: (batch_size, seq_len, dim)
        motion_latent: (batch_size, latent_dim0, latent_dim1)
        active_passive_bit: (batch_size,)
        """
        
        bs, seqlen, _ = alsn.shape
        # 
        active_passive_bit = active_passive_bit.to(torch.int)

        # 
        apb = self.active_passive_emb(active_passive_bit) 

        lsnid_emb = torch.IntTensor(lsn_id).to(spkemb.device)
        # breakpoint()
        lsnemb = self.lsn_id_emb(lsnid_emb).unsqueeze(1) # bs, 1, outdim 
        
        return spkemb, alsn, tlsn, apb, lsnemb



if __name__ == "__main__":
    bs = 4
    seq_len = 16
    lat0 = 1
    lat1 = 512
    out_dim = 512
    mm_emb_spk = torch.randn(bs, seq_len, out_dim)
    mm_emb_lsn = torch.randn(bs, seq_len, out_dim)
    motion_latent = torch.randn(bs, 2*lat0, lat1)
    active_passive_bit = [1, 0, 1, 0]
    fuser = TextAudioMotionFuser(None, out_dim)

    out = fuser(mm_emb_spk, mm_emb_lsn, motion_latent, active_passive_bit)
    print(out.shape)