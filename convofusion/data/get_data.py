from os.path import join as pjoin

import numpy as np
from .BEAT_DnD import BEATDataModule
from .utils import *



def get_collate_fn(name, phase="train", stage="vae_diffusion"):
    if name.lower() == "beatdnd":
        return beatdnd_vae_collate if stage == "vae" else beatdnd_collate
    else:
        raise NotImplementedError
    
# map config name to module&path
dataset_module_map = {
    "beatdnd": BEATDataModule,
}
#


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["beatdnd"]:
            # breakpoint()
            stage = cfg.TRAIN.STAGE
            if stage == "vae":
                data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            elif stage == "diffusion":
                data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase, stage=cfg.TRAIN.STAGE)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                # mean=mean,
                # std=std,
                # mean_eval=mean_eval,
                # std_eval=std_eval,
                motion_rep=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.POSE_REP"),
                dataset_path=data_root,
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
                sample_rate=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.SR"),
                num_mels=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.N_MELS"),
                hop_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.HOP_LEN"),
                fps=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.FPS"),
                face_joint_idx = eval(
                    f"cfg.DATASET.{dataset_name.upper()}.FACE_JOINT_IDX"),
                dataset_select=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.SELECT"),
                stage=cfg.TRAIN.STAGE,
            )
            datasets.append(dataset)
        else:
            raise NotImplementedError
    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
