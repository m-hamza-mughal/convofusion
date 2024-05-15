from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from convofusion.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from convofusion.models.operator import PositionalEncoding
from convofusion.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from convofusion.models.operator.position_encoding import build_position_encoding
from convofusion.utils.temos_utils import lengths_to_mask
"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""


class ConvoFusionVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.body_nfeats = 23*3
        self.hands_nfeats = 40*3
        self.arch = arch
        self.mlp_dist = ablation.MLP_DIST
        self.pe_type = ablation.PE_TYPE

        
        self.query_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos_decoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        

        body_encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        body_encoder_norm = nn.LayerNorm(self.latent_dim)
        self.body_encoder = SkipTransformerEncoder(body_encoder_layer, num_layers,
                                              body_encoder_norm)

        hands_encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        hands_encoder_norm = nn.LayerNorm(self.latent_dim)
        self.hands_encoder = SkipTransformerEncoder(hands_encoder_layer, num_layers,
                                              hands_encoder_norm)

        if self.arch == "all_encoder":
            body_decoder_norm = nn.LayerNorm(self.latent_dim)
            self.body_decoder = SkipTransformerEncoder(body_encoder_layer, num_layers,
                                                  body_decoder_norm)
            hands_decoder_norm = nn.LayerNorm(self.latent_dim)
            self.hands_decoder = SkipTransformerEncoder(hands_encoder_layer, num_layers,
                                                  hands_decoder_norm)
        elif self.arch == "encoder_decoder":
            body_decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            body_decoder_norm = nn.LayerNorm(self.latent_dim)
            self.body_decoder = SkipTransformerDecoder(body_decoder_layer, num_layers,
                                                  body_decoder_norm)
            hands_decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            hands_decoder_norm = nn.LayerNorm(self.latent_dim)
            self.hands_decoder = SkipTransformerDecoder(hands_decoder_layer, num_layers,
                                                  hands_decoder_norm)
        else:
            raise ValueError("Not support architecture!")


        # self.enc_body_connection = nn.Linear(self.latent_dim, self.latent_dim)
        # self.enc_hands_connection = nn.Linear(self.latent_dim, self.latent_dim)

        # self.dec_body_connection = nn.Linear(self.latent_dim, self.latent_dim)
        # self.dec_hands_connection = nn.Linear(self.latent_dim, self.latent_dim)

        if self.mlp_dist:
            self.body_global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.body_dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
            self.hands_global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.hands_dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            self.body_global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))
            self.hands_global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.body_skel_embedding = nn.Linear(self.body_nfeats, self.latent_dim)
        self.hands_skel_embedding = nn.Linear(self.hands_nfeats, self.latent_dim)
        self.body_final_layer = nn.Linear(self.latent_dim, self.body_nfeats)
        self.hands_final_layer = nn.Linear(self.latent_dim, self.hands_nfeats)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]
        # print(features.shape)
        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        # breakpoint()
        n_chunks = nframes // 16
        motion_feats = features.clone()
        motion_feats = motion_feats.reshape(bs*n_chunks, nframes//n_chunks, -1)

        # subtract the root position from each chunk
        # breakpoint()
        root_pos_init = motion_feats[:, :1, :3]
        root_pose_init_xz = root_pos_init * torch.tensor([1, 0, 1], device=device)
        motion_feats[:, :, :3] = motion_feats[:, :, :3] - root_pose_init_xz
        # breakpoint()
        mask = mask.reshape(bs*n_chunks, nframes//n_chunks)
        bs = bs*n_chunks

        body_feats = motion_feats[:, :, :self.body_nfeats] # [bs, nframes, body_nfeats]
        hands_feats = motion_feats[:, :, self.body_nfeats:] # [bs, nframes, hands_nfeats]

        xb = body_feats
        xh = hands_feats
        # Embed each human poses into latent vectors
        xb = self.body_skel_embedding(xb)
        xh = self.hands_skel_embedding(xh)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        xb = xb.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        xh = xh.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist_b = torch.tile(self.body_global_motion_token[:, None, :], (1, bs, 1))
        dist_h = torch.tile(self.hands_global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        distb_masks = torch.ones((bs, dist_b.shape[0]),
                                dtype=bool,
                                device=xb.device)
        body_aug_mask = torch.cat((distb_masks, mask), 1)

        disth_masks = torch.ones((bs, dist_h.shape[0]),
                                dtype=bool,
                                device=xh.device)
        hands_aug_mask = torch.cat((disth_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq_b = torch.cat((dist_b, xb), 0)
        xseq_h = torch.cat((dist_h, xh), 0)

        if self.pe_type == "convofusion":
            # breakpoint()
            xseq_b = self.query_pos_encoder(xseq_b)
            dist_b = self.body_encoder(xseq_b,
                                src_key_padding_mask=~body_aug_mask)[:dist_b.shape[0]]
            xseq_h = self.query_pos_encoder(xseq_h)
            dist_h = self.hands_encoder(xseq_h,
                                src_key_padding_mask=~hands_aug_mask)[:dist_h.shape[0]]
        else:
            raise ValueError("Not support position encoding type!")

        # make body and hands connections here. 
        # dist_b_out = dist_b + self.enc_hands_connection(dist_h)
        # dist_h_out = dist_h + self.enc_body_connection(dist_b)
        dist_b_out = dist_b
        dist_h_out = dist_h

        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.mlp_dist:
            body_tokens_dist = self.body_dist_layer(dist_b_out)
            b_mu = body_tokens_dist[:, :, :self.latent_dim]
            b_logvar = body_tokens_dist[:, :, self.latent_dim:]

            hands_tokens_dist = self.hands_dist_layer(dist_h_out)
            h_mu = hands_tokens_dist[:, :, :self.latent_dim]
            h_logvar = hands_tokens_dist[:, :, self.latent_dim:]
        else:
            b_mu = dist_b_out[0:self.latent_size, ...]
            b_logvar = dist_b_out[self.latent_size:, ...]
            h_mu = dist_h_out[0:self.latent_size, ...]
            h_logvar = dist_h_out[self.latent_size:, ...]

        # print("b_mu shape", b_mu.shape, "h_mu shape", h_mu.shape)
        mu = torch.cat((b_mu, h_mu), axis=0) # [latent_size*2, bs, latent_dim]
        logvar = torch.cat((b_logvar, h_logvar), axis=0)

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        # print("latent shape", latent.shape)
        # breakpoint()
        latent = latent.reshape(-1, bs // n_chunks, n_chunks, self.latent_dim)
        return latent, dist , motion_feats.reshape(bs//n_chunks, motion_feats.shape[1]*n_chunks, -1)

    def decode(self, z: Tensor, lengths: List[int]):
        # print(z.shape)
        _, bs, n_chunks, _ = z.shape
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        # nframes = nframes // 8
        # bs = bs * 8

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        # breakpoint()
        z_b, z_h = torch.chunk(z, 2, dim=0)
        # breakpoint()
        z_b = z_b.squeeze(0)
        z_b = z_b.permute(1, 0, 2)

        z_h = z_h.squeeze(0)
        z_h = z_h.permute(1, 0, 2)
        # breakpoint()
        # make body and hands connections here.
        # z_body = z_body + self.dec_hands_connection(z_hands)
        # z_hands = z_hands + self.dec_body_connection(z_body)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq_b = torch.cat((z_b, queries), axis=0)
            z_mask_b = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=zb.device)
            augmask_b = torch.cat((z_mask_b, mask), axis=1)

            xseq_h = torch.cat((z_h, queries), axis=0)
            z_mask_h = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=zh.device)
            augmask_h = torch.cat((z_mask_h, mask), axis=1)

            if self.pe_type == "convofusion":
                xseq_b = self.query_pos_decoder(xseq_b)
                output_b = self.body_decoder(
                    xseq_b, src_key_padding_mask=~augmask_b)[z_b.shape[0]:]
                xseq_h = self.query_pos_decoder(xseq_h)
                output_h = self.hands_decoder(
                    xseq_h, src_key_padding_mask=~augmask_h)[z_h.shape[0]:]
            else:
                raise ValueError("Not support position encoding type!")

        elif self.arch == "encoder_decoder":
            if self.pe_type == "convofusion":
                queries = self.query_pos_decoder(queries)
                z_b = self.mem_pos_decoder(z_b)
                output_b = self.body_decoder(
                    tgt=queries,
                    memory=z_b,
                    tgt_key_padding_mask=~mask,
                    # query_pos=query_pos,
                    # pos=mem_pos,
                ).squeeze(0)

                z_h = self.mem_pos_decoder(z_h)
                output_h = self.hands_decoder(
                    tgt=queries,
                    memory=z_h,
                    tgt_key_padding_mask=~mask,
                    # query_pos=query_pos,
                    # pos=mem_pos,
                ).squeeze(0)

                # query_pos = self.query_pos_decoder(queries)
                # # mem_pos = self.mem_pos_decoder(z)
                # output = self.decoder(
                #     tgt=queries,
                #     memory=z,
                #     tgt_key_padding_mask=~mask,
                #     query_pos=query_pos,
                #     # pos=mem_pos,
                # ).squeeze(0)
            else:
                raise ValueError("Not support position encoding type!")

        output_b = self.body_final_layer(output_b) # [nframes, bs, body_nfeats]
        output_h = self.hands_final_layer(output_h) # [nframes, bs, hands_nfeats]

        output = torch.cat((output_b, output_h), axis=-1) # [nframes, bs, nfeats]

        
        # add back the root position of previous frame to each start of each chunk
        # for chunk_idx in range()

        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        # breakpoint()
        
        
        # chunk_len = nframes // n_chunks
        # for chunk_idx in range(1, n_chunks):
        #     feats[:, chunk_idx*chunk_len:(chunk_idx+1)*chunk_len, :3] += feats[:, [(chunk_idx*chunk_len)-1], :3] * torch.tensor([1, 0, 1], device=feats.device)
        
        return feats
