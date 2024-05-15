# Taken from attend-and-excite codebase.
# and modified for word excitation guidance.
# https://github.com/yuval-alaluf/Attend-and-Excite

import torch
import torch.nn.functional as F

from convofusion.models.operator import GaussianSmoothing


def aggregate_attentions(att_mats):
        # Aggregates the attention across the different layers
        att_mats = torch.mean(att_mats, dim=1) # (bs, layers, motion_seq_len, text_seq_len)
        return att_mats

def get_max_attention_at_indices(att_mat, batch_idxs, smooth_attentions=False, normalize_eot=False, eot_indices=[]):
    # Get the maximum attention value for each token in the focus indices
    # breakpoint()
    # remove eos and bos tokens from the attention matrix
    last_idx = -1
    
    # breakpoint()
    if normalize_eot:
        assert len(eot_indices) > 0, "Need to provide eot indices for normalization"
        assert att_mat.shape[0] == 1, "EOS/BOS normalization only works for test batch size 1 currently"
        last_idx = eot_indices[0]
    # print(f"Last index: {last_idx}")
    attention_for_text = att_mat[:, :, 1:last_idx]
    # 
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

    # 
    smoothing_operator = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2)
    if smooth_attentions:
        input = F.pad(attention_for_text.unsqueeze(1), (1, 1, 1, 1), mode='reflect')
        attention_for_text = smoothing_operator(input).squeeze(1)
    # 

    batch_max_indices_list = []
    for b_i in range(len(batch_idxs)):
        max_indices_list = []
        if len(batch_idxs[b_i]) == 0:
            batch_max_indices_list.append(max_indices_list)
            continue
        for i in batch_idxs[b_i]:
            motion_chunk = attention_for_text[b_i, :, i-1] # -1 because we removed the bos token
            
            #
            max_indices_list.append(motion_chunk.max(dim=-1)[0]) # choose only max values and discard argmax
        batch_max_indices_list.append(max_indices_list)
    return batch_max_indices_list 


def update_latent(latents, loss, lr):
    """ Update the latent according to the computed loss.
    Taken from attend-and-excite codebase.
    https://github.com/yuval-alaluf/Attend-and-Excite
    """
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
    latents = latents - lr * grad_cond
    return latents


def compute_attention_focus_loss(max_attention_at_indices):
    " the attend-and-excite loss using the maximum attention value for each token - extended for batch"
    losses = []
    for sample in max_attention_at_indices:
        token_losses = []
        if len(sample) == 0:
            losses.append(torch.tensor(0.).cuda())
            continue
        for token in sample:
            token_losses.append(torch.max(torch.zeros_like(token), 1. - token))
        # losses.append(torch.max(torch.stack(token_losses), dim=-1)[0])
        losses.append(torch.mean(torch.stack(token_losses), dim=-1))

    if len(losses) == 0:
        return torch.tensor(0.).cuda(), None

    losses = torch.stack(losses, dim=-1)
    loss = torch.mean(losses) # mean across batch
    return loss, losses