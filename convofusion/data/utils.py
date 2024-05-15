import torch


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas



# an adapter to our collate func
def beatdnd_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[1], reverse=True)

    adapted_batch = {
        "motion_spk":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "length": [b[1] for b in notnone_batches],
        "motion_lsn": 
        collate_tensors([torch.tensor(b[2]).float() for b in notnone_batches]),
        "melspec_spk": 
        collate_tensors([torch.tensor(b[3]).float() for b in notnone_batches]),
        "melspec_lsn":
        collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "audio_spk":
        collate_tensors([torch.tensor(b[5]).float() for b in notnone_batches]),
        "audio_lsn":
        collate_tensors([torch.tensor(b[6]).float() for b in notnone_batches]),
        "text_spk":
        [b[7] for b in notnone_batches],
        "text_lsn":
        [b[8] for b in notnone_batches],
        "active_passive_lsn":
        collate_tensors([torch.tensor(b[9]).float() for b in notnone_batches]), 
        "name":
        [b[10] for b in notnone_batches],
        "spk_name": [b[11] for b in notnone_batches],
        "lsn_name": [b[12] for b in notnone_batches],
        "lsn_id": [b[13] for b in notnone_batches],
        "other_mlsn": [b[14] for b in notnone_batches],
        "combined_audio":
        collate_tensors([torch.tensor(b[15]).float() for b in notnone_batches]),
        "seg_lsn": 
        [b[16] for b in notnone_batches],
        "seg_spk":
        [b[17] for b in notnone_batches],
        "sem_lsn":
        collate_tensors([torch.tensor(b[18]).float() for b in notnone_batches]),
        'sem_info': [b[19] for b in notnone_batches],

    }
    return adapted_batch

def beatdnd_vae_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[1], reverse=True)

    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "length": [b[1] for b in notnone_batches],
        "name": [b[2] for b in notnone_batches],
    }
    return adapted_batch

