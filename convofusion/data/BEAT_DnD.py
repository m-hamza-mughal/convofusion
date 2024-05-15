import numpy as np
import torch

from convofusion.data.beat_dnd.utils.motion_rep_utils import convert_6D_to_euler, convert_euler_to_6D, forward_kinematics_cont6d

from .base import BASEDataModule
from .beat_dnd.dataset import BEATAugReactionDataset, MotionDataset 


class BEATDataModule(BASEDataModule):
    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "beatdnd"
        self.stage = cfg.TRAIN.STAGE
        if self.stage == "vae":
            self.Dataset = MotionDataset
        else:
            self.Dataset = BEATAugReactionDataset
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "debug": True,
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        self.njoints = self._sample_set.njoints
        self.batch_size = batch_size

    def euler2rep6d(self, features):
        return convert_euler_to_6D(features, self.njoints)

    def rep6d2euler(self, features):
        return convert_6D_to_euler(features, self.njoints)

    # ADD FK CODE HERE from 6d to joints
    def rep6d2joints(self, features_batch):
        # features_batch shape (batch_size, seq_len, 3 + 6*njoints)
        # 
        kinematic_tree = self.cfg.DATASET.BEATDND.KINEMATIC_TREE
        offset = np.load(self.cfg.DATASET.BEATDND.OFFSET_NPY_PATH)
        offset = torch.from_numpy(offset).float().unsqueeze(0)

        # batch_size, seq_len, _ = features_batch.shape
        features_batch = features_batch.view(-1, 3 + 6 * self.njoints) # (batch_size*seq_len, 3 + 6*njoints)
        
        root_pos = features_batch[:, :3] # (batch_size*seq_len, 3)
        cont6d_params = features_batch[:, 3:].view(-1, self.njoints, 6) # (batch_size*seq_len, njoints, 6)
        joints = forward_kinematics_cont6d(cont6d_params, root_pos, offset, kinematic_tree) # (batch_size*seq_len, njoints, 3)
        #
        return joints

