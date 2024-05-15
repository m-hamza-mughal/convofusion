# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional
from numpy import block

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x

class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = decoder_layer.d_model
        
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        x = tgt

        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, cond_params, time_embed,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                causal_attn=False):
        output = tgt

        intermediate = []
        att_mats_per_layer = []
        # condparam_per_layer = []
        for layer in self.layers:
            output, att_mats = layer(output, memory, cond_params, time_embed, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, causal_attn=causal_attn)
            # breakpoint()
            if len(att_mats_per_layer) == 0:
                att_mats_per_layer = [[] for _ in att_mats]
            
            for i, att_mat in enumerate(att_mats):
                att_mats_per_layer[i].append(att_mat)
            # breakpoint()
            # condparam_per_layer.append(cond_param)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # breakpoint()
        att_mats = [torch.stack(att_mats).permute(1, 0, 2, 3) for att_mats in att_mats_per_layer]
        # breakpoint()
        # cond_param = torch.stack(condparam_per_layer)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0), att_mats#, cond_param


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
                     
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class WeightedSumFuser(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.rand(out_dim, in_dim))

    def forward(self, x, K):
        # breakpoint()
        weight_chunks = torch.chunk(self.W, len(K), dim=-1)
        weight = torch.cat([weight_chunk * K[i] for i, weight_chunk in enumerate(weight_chunks)], dim=-1)

        return nn.functional.linear(x, weight, bias=None)

class TimeBlock(nn.Module):

    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, h, emb):
        """
        h: T, B, D
        emb: 1, B, D
        """
        # 1, B, 2D
        # breakpoint()
        emb_out = self.emb_layers(emb)
        # scale: 1, B, D / shift: 1, B, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        # print(h.shape, emb.shape, scale.shape, shift.shape)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class TransformerDecoderLayer2Att(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # breakpoint()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_block1 = TimeBlock(d_model, dropout)

        self.multihead_attn_spkemb = nn.MultiheadAttention(d_model, 1, dropout=dropout)
        # self.multihead_attn_aspk = nn.MultiheadAttention(d_model, 1, dropout=dropout)
        # self.multihead_attn_tspk = nn.MultiheadAttention(d_model, 1, dropout=dropout)

        self.multihead_attn_tlsn = nn.MultiheadAttention(d_model, 1, dropout=dropout)
        self.multihead_attn_alsn = nn.MultiheadAttention(d_model, 1, dropout=dropout)

        self.multihead_attn_apb = nn.MultiheadAttention(d_model, 1, dropout=dropout)
        self.multihead_attn_lsnemb = nn.MultiheadAttention(d_model, 1, dropout=dropout)

        # --- JOINED CROSSATTENTION
        # self.multihead_attn_joined = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # self.att_fuser = WeightedSumFuser(d_model*5, d_model)
        self.att_fuser = nn.Linear(d_model*5, d_model)
        self.time_block2 = TimeBlock(d_model, dropout)

        # Implementation of Feedforward model
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.spkemb_norm = nn.LayerNorm(d_model)
        self.alsn_norm = nn.LayerNorm(d_model)
        self.tlsn_norm = nn.LayerNorm(d_model)
        self.apb_norm = nn.LayerNorm(d_model)
        self.lsnemb_norm = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # self.attention_weight = nn.Parameter(torch.rand(1))

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, cond_params, time_embed,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     causal_attn=False):
                     
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        spkemb, alsn, tlsn, apb, lsnemb  = memory
        # spkemb = spkemb.permute(2, 1, 0)
        tgt2_spkemb, att_spkemb = self.multihead_attn_spkemb(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(spkemb, pos),
                                   value=spkemb, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        # tgt2_aspk, att_aspk = self.multihead_attn_aspk(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
        #                            key=self.with_pos_embed(aspk, pos),
        #                            value=aspk, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)
        # tgt2_tspk, att_tspk = self.multihead_attn_tspk(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
        #                            key=self.with_pos_embed(tspk, pos),
        #                            value=tspk, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)

        tgt2_alsn, att_alsn = self.multihead_attn_alsn(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(alsn, pos),
                                   value=alsn, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2_tlsn, att_tlsn = self.multihead_attn_tlsn(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(tlsn, pos),
                                   value=tlsn, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt2_apb, att_apb = self.multihead_attn_apb(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(apb, pos),
                                   value=apb, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2_lsnemb, att_lsnemb = self.multihead_attn_lsnemb(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(lsnemb, pos),
                                   value=lsnemb, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt2_cat = torch.cat([tgt2_spkemb, tgt2_alsn, tgt2_tlsn, tgt2_apb, tgt2_lsnemb], dim=-1)

        tgt2 = self.att_fuser(tgt2_cat)     
        
        
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, (att_spkemb, att_alsn, att_tlsn, att_apb, att_lsnemb)

    def forward_pre(self, tgt, memory, cond_params, time_embed,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[dict] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    causal_attn=False):
        
        # breakpoint()
        
        #self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # time embedding block
        tgt = tgt + self.time_block1(tgt, time_embed)

        # multihead attention
        tgt2 = self.norm2(tgt)
        spkemb, alsn, tlsn, apb, lsnemb  = memory
        # breakpoint()
        spkemb = self.spkemb_norm(spkemb)
        alsn = self.alsn_norm(alsn)
        tlsn = self.tlsn_norm(tlsn)
        apb = self.apb_norm(apb)
        lsnemb = self.lsnemb_norm(lsnemb)
        # breakpoint()
        spkemb_padmask = memory_key_padding_mask.get('spkemb')
        alsn_padmask = memory_key_padding_mask.get('alsn')
        tlsn_padmask = memory_key_padding_mask.get('tlsn')
        apb_padmask = memory_key_padding_mask.get('apb')
        lsnemb_padmask = memory_key_padding_mask.get('lsnemb')
        # breakpoint()
        tgt2_spkemb, att_spkemb = self.multihead_attn_spkemb(query=self.with_pos_embed(tgt2, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(spkemb, pos),
                                   value=spkemb, attn_mask=memory_mask,
                                   key_padding_mask=spkemb_padmask, 
                                   is_causal=causal_attn)
        # tgt2_aspk, att_aspk = self.multihead_attn_aspk(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
        #                            key=self.with_pos_embed(aspk, pos),
        #                            value=aspk, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)
        # tgt2_tspk, att_tspk = self.multihead_attn_tspk(query=self.with_pos_embed(tgt, query_pos), # Todo: will need to change dim of pos
        #                            key=self.with_pos_embed(tspk, pos),
        #                            value=tspk, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)

        tgt2_alsn, att_alsn = self.multihead_attn_alsn(query=self.with_pos_embed(tgt2, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(alsn, pos),
                                   value=alsn, attn_mask=memory_mask,
                                   key_padding_mask=alsn_padmask,
                                   is_causal=causal_attn)
        tgt2_tlsn, att_tlsn = self.multihead_attn_tlsn(query=self.with_pos_embed(tgt2, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(tlsn, pos),
                                   value=tlsn, attn_mask=memory_mask,
                                   key_padding_mask=tlsn_padmask,
                                   is_causal=causal_attn)

        tgt2_apb, att_apb = self.multihead_attn_apb(query=self.with_pos_embed(tgt2, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(apb, pos),
                                   value=apb, attn_mask=memory_mask,
                                   key_padding_mask=apb_padmask,
                                   is_causal=causal_attn)
        tgt2_lsnemb, att_lsnemb = self.multihead_attn_lsnemb(query=self.with_pos_embed(tgt2, query_pos), # Todo: will need to change dim of pos
                                   key=self.with_pos_embed(lsnemb, pos),
                                   value=lsnemb, attn_mask=memory_mask,
                                   key_padding_mask=lsnemb_padmask)
        # breakpoint()
        # tgt2_cat = torch.cat([tgt2_spkemb, tgt2_aspk, tgt2_tspk, tgt2_alsn, tgt2_tlsn, tgt2_apb, tgt2_lsnemb], dim=-1)
        tgt2_cat = torch.cat([tgt2_spkemb, tgt2_alsn, tgt2_tlsn, tgt2_apb, tgt2_lsnemb], dim=-1)
        # breakpoint()
        
        # tgt2 = cond_params[0] * tgt2_spkemb + cond_params[1] * tgt2_alsn + cond_params[2] * tgt2_tlsn + cond_params[3] * tgt2_apb + cond_params[4] * tgt2_lsnemb
        # breakpoint()
        tgt2 = self.att_fuser(tgt2_cat) #, cond_params)

        # --- JOINED CROSSATTENTION
        # lsnemb = lsnemb.repeat(8, 1, 1) # optional step to help lsnemb not get ignored by attention. 
        # joined_cond = torch.cat([spkemb, alsn, tlsn, apb, lsnemb], dim=0)
        # spkemb_pos = self.with_pos_embed(spkemb, pos)
        # alsn_pos = self.with_pos_embed(alsn, pos)
        # tlsn_pos = self.with_pos_embed(tlsn, pos)
        # apb_pos = self.with_pos_embed(apb, pos)
        # lsnemb_pos = self.with_pos_embed(lsnemb, pos)
        # joined_cond_pos = torch.cat([spkemb_pos, alsn_pos, tlsn_pos, apb_pos, lsnemb_pos], dim=0)
        # TODO : add mask for text + others
        # tgt2, att_joined = self.multihead_attn_joined(query=self.with_pos_embed(tgt2, query_pos), # Todo: will need to change dim of pos
        #                            key=joined_cond_pos,
        #                            value=joined_cond, attn_mask=None,
        #                            key_padding_mask=None, 
        #                            is_causal=False)

        tgt = tgt + self.dropout2(tgt2)

        # time embedding block
        tgt = tgt + self.time_block2(tgt, time_embed)

        # last linear layers
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        # breakpoint()
        # return tgt, (att_spkemb, att_aspk, att_tspk, att_alsn, att_tlsn, att_apb, att_lsnemb)
        return tgt, (att_spkemb, att_alsn, att_tlsn, att_apb, att_lsnemb)#, self.cond_params
        # --- JOINED CROSSATTENTION
        # return tgt, (att_joined,)

    def forward(self, tgt, memory, cond_params, time_embed,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[dict] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                causal_attn=False):
        # breakpoint()
        if self.normalize_before:
            return self.forward_pre(tgt, memory, cond_params,time_embed, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, causal_attn)
        return self.forward_post(tgt, memory, cond_params, time_embed, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, causal_attn)


def _get_clone(module):
    return copy.deepcopy(module)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")