import torch
import torch.nn as nn
from torch import  nn
from convofusion.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from convofusion.models.operator import PositionalEncoding
from convofusion.models.operator.cross_attention import (SkipTransformerEncoder,
                                                 TransformerDecoder,
                                                 TransformerDecoderLayer2Att,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer)
from convofusion.models.operator.position_encoding import build_position_encoding
from convofusion.utils.temos_utils import lengths_to_mask


class Denoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768, 
                 audio_encoded_dim: int = 512,
                 nclasses: int = 10,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim
        self.audio_encoded_dim = audio_encoded_dim
        self.condition = condition
        self.abl_plus = True #changed
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE
        self.causal_attn = ablation.CAUSAL_ATTN

        if self.diffusion_only:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)
        else:
            self.latent_embd = nn.Linear(latent_dim[-1], text_encoded_dim) #if latent_dim[-1] != text_encoded_dim else nn.Identity()
            self.latent_proj = nn.Linear(text_encoded_dim, latent_dim[-1]) #if text_encoded_dim != latent_dim[-1] else nn.Identity()

        
        # emb proj
        # breakpoint()
        if self.condition in ["text", "text_uncond"]: 
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                    self.latent_dim)
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))

        elif self.condition in ["text+audio", "textaudio_uncond"]:
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                    text_encoded_dim)
            # project time+text to latent_dim
            # if text_encoded_dim != self.latent_dim:
            #     # todo 10.24 debug why relu
            #     self.txt_emb_proj = nn.Sequential(
            #         nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
            
            # text+audio condition
            # project time from text_encoded_dim to latent_dim
            # self.time_proj = Timesteps(audio_encoded_dim, flip_sin_to_cos,
            #                             freq_shift)
            # self.time_embedding = TimestepEmbedding(audio_encoded_dim,
            #                                         self.latent_dim)
            # project time+text to latent_dim
            # if audio_encoded_dim != self.latent_dim:
            #     # todo 10.24 debug why relu
            #     self.audio_emb_proj = nn.Sequential(
            #         nn.ReLU(), nn.Linear(audio_encoded_dim, self.latent_dim))

        elif self.condition in ['action']:
            self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(self.latent_dim,
                                                    self.latent_dim)
            self.emb_proj = EmbedAction(nclasses,
                                        self.latent_dim,
                                        guidance_scale=guidance_scale,
                                        guidance_uncodp=guidance_uncondp)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        if self.pe_type == "convofusion":
            self.query_pos = build_position_encoding(
                text_encoded_dim, position_embedding="sine_bh") # HARD_CODED sine_bh
            self.mem_pos = build_position_encoding(
                text_encoded_dim, position_embedding=position_embedding)
            # self.mem_mpsk_pos = build_position_encoding(
            #     text_encoded_dim, position_embedding="sine_bh") # HARD_CODED
        else:
            raise ValueError("Not Support PE type")

        self.bh_embedding = nn.Embedding(2, text_encoded_dim)
        self.condition_embedding = nn.Embedding(5, text_encoded_dim)
        

        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)
        elif self.arch == "trans_dec":
            self.cond_params = nn.Parameter(1/5*torch.ones(5))
            decoder_layer = TransformerDecoderLayer2Att(
                text_encoded_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(text_encoded_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                mem_mask_dict=dict(),
                **kwargs):
        # 0.  dimension matching
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        # breakpoint()
        sample = sample.permute(1, 0, 2) # ntokens(8*2), bs, dim 

        # breakpoint()
        if not self.diffusion_only:
            sample = self.latent_embd(sample)  # dim -> text_encoded_dim

        # 0. check lengths for no vae (diffusion only)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        # breakpoint()
        if self.condition in ["text", "text_uncond"]:
            # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
            # textembedding projection
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)

        elif self.condition in ["text+audio", "textaudio_uncond"]:
            # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
            # spk_emb, aspk, tspk, alsn, tlsn, apb, lsnemb = encoder_hidden_states
            spk_emb, alsn, tlsn, apb, lsnemb = encoder_hidden_states
            # breakpoint()
            spk_emb = spk_emb.permute(1, 0, 2) # lat1, bs, lat0
            # aspk = aspk.permute(1, 0, 2) # seq_len, bs, enc_dim
            # tspk = tspk.permute(1, 0, 2) # seq_len, bs, enc_dim
            alsn = alsn.permute(1, 0, 2) # seq_len, bs, enc_dim
            tlsn = tlsn.permute(1, 0, 2) # seq_len, bs, enc_dim
            apb = apb.permute(1, 0, 2) # 1, bs, enc_dim
            lsnemb = lsnemb.permute(1, 0, 2) # 1, bs, enc_dim

            # # text_emb = text_encoded  # [num_words, bs, latent_dim]
            # # textembedding projection
            # if self.text_encoded_dim != self.latent_dim:
            #     # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
            #     tspk = self.txt_emb_proj(tspk)
            #     tlsn = self.txt_emb_proj(tlsn)
            # else:
            #     tspk = tspk
            #     tlsn = tlsn

            # # audio_emb = audio_encoded  # [num_words, bs, latent_dim]
            # # audioembedding projection
            # if self.audio_encoded_dim != self.latent_dim:
            #     # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
            #     aspk = self.audio_emb_proj(tspk)
            #     alsn = self.audio_emb_proj(tlsn)
            # else:
            #     aspk = aspk
            #     alsn = alsn

            # breakpoint()
            if self.abl_plus:
                # breakpoint()
                # tspk = time_emb + tspk
                tlsn = time_emb + tlsn
                # aspk = time_emb + aspk
                alsn = time_emb + alsn

                spk_emb = time_emb + spk_emb
                apb = time_emb + apb
                lsnemb = time_emb + lsnemb
            else:
                # tspk = torch.cat((time_emb, tspk), 0)
                tlsn = torch.cat((time_emb, tlsn), 0)
                # aspk = torch.cat((time_emb, aspk), 0)
                alsn = torch.cat((time_emb, alsn), 0)

                spk_emb = torch.cat((time_emb, spk_emb), 0)
                apb = torch.cat((time_emb, apb), 0)
                lsnemb = torch.cat((time_emb, lsnemb), 0)
                
            
        elif self.condition in ['action']:
            action_emb = self.emb_proj(encoder_hidden_states)
            if self.abl_plus:
                emb_latent = action_emb + time_emb
            else:
                emb_latent = torch.cat((time_emb, action_emb), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # 4. transformer
        if self.arch == "trans_enc":
            if self.diffusion_only:
                sample = self.pose_embd(sample)
                xseq = torch.cat((spk_emb, aspk, tspk, alsn, tlsn, apb, lsnemb, sample), axis=0)
            else:
                xseq = torch.cat((sample, spk_emb, aspk, tspk, alsn, tlsn, apb, lsnemb), axis=0)

            # if self.ablation_skip_connection:
            #     xseq = self.query_pos(xseq)
            #     tokens = self.encoder(xseq)
            # else:
            #     # adding the timestep embed
            #     # [seqlen+1, bs, d]
            #     # todo change to query_pos_decoder
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq)

            if self.diffusion_only:
                sample = tokens[spk_emb.shape[0] + aspk.shape[0] + tspk.shape[0] + alsn.shape[0] + tlsn.shape[0] + apb.shape[0] + lsnemb.shape[0]:]
                sample = self.pose_proj(sample)

                # zero for padded area
                sample[~mask.T] = 0
            else:
                sample = tokens[:sample.shape[0]]

        elif self.arch == "trans_dec":
            if self.diffusion_only:
                sample = self.pose_embd(sample)

            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            # breakpoint()
            body_embed = self.bh_embedding(torch.IntTensor([0]).to(sample.device))
            hands_embed = self.bh_embedding(torch.IntTensor([1]).to(sample.device))

            bs = sample.shape[1]
            hand_idxs = hands_embed[:, None, :].repeat(1, bs, 1)
            body_idxs = body_embed[:, None, :].repeat(1, bs, 1)

            sample[0::2] = sample[0::2] + body_idxs
            sample[1::2] = sample[1::2] + hand_idxs
            # breakpoint()
            sample = self.query_pos(sample)
            # print(emb_latent.shape)
            # spk_emb, aspk, tspk, alsn, tlsn, apb, lsnemb = \
            #     self.mem_pos(spk_emb), self.mem_pos(aspk), self.mem_pos(tspk), self.mem_pos(alsn), \
            #     self.mem_pos(tlsn), self.mem_pos(apb), self.mem_pos(lsnemb)

            spkemb_num = self.condition_embedding(torch.IntTensor([0]).to(sample.device))
            spkemb_num = spkemb_num[:, None, :].repeat(1, bs, 1)
            spk_emb = spk_emb + spkemb_num

            alsn_num = self.condition_embedding(torch.IntTensor([1]).to(sample.device))
            alsn_num = alsn_num[:, None, :].repeat(1, bs, 1)
            alsn = alsn + alsn_num

            tlsn_num = self.condition_embedding(torch.IntTensor([2]).to(sample.device))
            tlsn_num = tlsn_num[:, None, :].repeat(1, bs, 1)
            tlsn = tlsn + tlsn_num

            apb_num = self.condition_embedding(torch.IntTensor([3]).to(sample.device))
            apb_num = apb_num[:, None, :].repeat(1, bs, 1)
            apb = apb + apb_num

            lsnid_num = self.condition_embedding(torch.IntTensor([4]).to(sample.device))
            lsnid_num = lsnid_num[:, None, :].repeat(1, bs, 1)
            lsnemb = lsnemb + lsnid_num
            
            alsn, tlsn, apb, lsnemb, spk_emb = \
                self.mem_pos(alsn), self.mem_pos(tlsn), self.mem_pos(apb), self.mem_pos(lsnemb), self.mem_pos(spk_emb)

            # mspk[0::2] = mspk[0::2] + body_idxs
            # mspk[1::2] = mspk[1::2] + hand_idxs
            # mspk = self.mem_mpsk_pos(mspk)
            # breakpoint()
            # sample, att_mats = self.decoder(tgt=sample, memory=[mspk, aspk, tspk, alsn, tlsn, apb, lsnemb])
            # breakpoint()
            sample, att_mats = self.decoder(tgt=sample, 
                                            memory=[spk_emb, alsn, tlsn, apb, lsnemb], 
                                            cond_params=self.cond_params, 
                                            time_embed=time_emb, 
                                            memory_key_padding_mask=mem_mask_dict, 
                                            causal_attn=self.causal_attn
                                        )
            # breakpoint()

            # breakpoint()
            sample = sample.squeeze(0)

            if self.diffusion_only:
                sample = self.pose_proj(sample)
                # zero for padded area
                sample[~mask.T] = 0
        else:
            raise TypeError("{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        if not self.diffusion_only:
            sample = self.latent_proj(sample)  # text_encoded_dim -> latent_dim
        
        sample = sample.permute(1, 0, 2)

        return (sample, att_mats)


class EmbedAction(nn.Module):

    def __init__(self,
                 num_actions,
                 latent_dim,
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        if not self.training and self.guidance_scale > 1.0:
            uncond, output = output.chunk(2)
            uncond_out = self.mask_cond(uncond, force=True)
            out = self.mask_cond(output)
            output = torch.cat((uncond_out, out))

        output = self.mask_cond(output)

        return output.unsqueeze(0)

    def mask_cond(self, output, force=False):
        bs, d = output.shape
        # classifer guidence
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return output * (1. - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
