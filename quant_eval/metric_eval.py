import librosa
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.pyplot import figure
import math
from scipy.signal import argrelextrema
from motion_autoencoder import HalfEmbeddingNet
from collections import OrderedDict
from quaternion import qbetween_np, qrot_np
from scipy import linalg
import argparse


from tqdm import tqdm
import torch


class FIDCalculator(object):
    @staticmethod
    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = FIDCalculator.calculate_frechet_distance(
                A_mu, A_sigma, B_mu, B_sigma
            )
        except ValueError:
            frechet_dist = 1e10
        return frechet_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                    representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                    representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Alignment(object):
    """Class to calculate alignment between audio and pose data"""

    def __init__(self, sigma, order):
        self.sigma = sigma
        self.order = order
        self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []

    def load_audio(self, wave, t_start, t_end, without_file=False, sr_audio=16000):
        if without_file:
            y = wave
            sr = sr_audio
        else:
            y, sr = librosa.load(wave, sr=sr_audio)
        short_y = y  # [int(t_start*sr):int(t_end*sr)]
        self.oenv = librosa.onset.onset_strength(y=short_y, sr=sr)
        self.times = librosa.times_like(self.oenv)
        # Detect events without backtracking
        onset_raw = librosa.onset.onset_detect(
            onset_envelope=self.oenv, backtrack=False
        )
        if len(onset_raw) == 0:
            # print(len(wave))
            return None, None, None
        onset_bt = librosa.onset.onset_backtrack(onset_raw, self.oenv)
        self.S = np.abs(librosa.stft(y=short_y))
        self.rms = librosa.feature.rms(S=self.S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, self.rms[0])
        return onset_raw, onset_bt, onset_bt_rms

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        

        data_each_file = pose  # .reshape(-1, 189//3, 3)
        vel = data_each_file[1:, :] - data_each_file[:-1, :]
        
        # l2
        # breakpoint()
        vel_right_shoulder = np.linalg.norm(
            np.array([vel[:, 9 * 3], vel[:, 9 * 3 + 1], vel[:, 9 * 3 + 2]]), axis=0
        )
        vel_right_arm = np.linalg.norm(
            np.array([vel[:, 10 * 3], vel[:, 10 * 3 + 1], vel[:, 10 * 3 + 2]]), axis=0
        )
        vel_right_wrist = np.linalg.norm(
            np.array([vel[:, 11 * 3], vel[:, 11 * 3 + 1], vel[:, 11 * 3 + 2]]), axis=0
        )
        beat_right_arm = argrelextrema(vel_right_arm, np.less, order=self.order)
        beat_right_shoulder = argrelextrema(
            vel_right_shoulder, np.less, order=self.order
        )
        beat_right_wrist = argrelextrema(vel_right_wrist, np.less, order=self.order)
        vel_left_shoulder = np.linalg.norm(
            np.array([vel[:, 5 * 3], vel[:, 5 * 3 + 1], vel[:, 5 * 3 + 2]]), axis=0
        )
        vel_left_arm = np.linalg.norm(
            np.array([vel[:, 6 * 3], vel[:, 6 * 3 + 1], vel[:, 6 * 3 + 2]]), axis=0
        )
        vel_left_wrist = np.linalg.norm(
            np.array([vel[:, 7 * 3], vel[:, 7 * 3 + 1], vel[:, 7 * 3 + 2]]), axis=0
        )
        beat_left_arm = argrelextrema(vel_left_arm, np.less, order=self.order)
        beat_left_shoulder = argrelextrema(vel_left_shoulder, np.less, order=self.order)
        beat_left_wrist = argrelextrema(vel_left_wrist, np.less, order=self.order)
        return (
            beat_right_arm,
            beat_right_shoulder,
            beat_right_wrist,
            beat_left_arm,
            beat_left_shoulder,
            beat_left_wrist,
        )

    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        (
            beat_right_arm,
            beat_right_shoulder,
            beat_right_wrist,
            beat_left_arm,
            beat_left_shoulder,
            beat_left_wrist,
        ) = self.load_pose(pose, t_start, t_end, pose_fps)
        return (
            onset_raw,
            onset_bt,
            onset_bt_rms,
            beat_right_arm,
            beat_right_shoulder,
            beat_right_wrist,
            beat_left_arm,
            beat_left_shoulder,
            beat_left_wrist,
        )

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            (
                beat_right_arm,
                beat_right_shoulder,
                beat_right_wrist,
                beat_left_arm,
                beat_left_shoulder,
                beat_left_wrist,
            ) = self.load_pose(pose, i, i + dur, pose_fps)
            dis_all_b2a = self.calculate_align(
                onset_raw,
                onset_bt,
                onset_bt_rms,
                beat_right_arm,
                beat_right_shoulder,
                beat_right_wrist,
                beat_left_arm,
                beat_left_shoulder,
                beat_left_wrist,
            )
            print(f"{i}s: ", dis_all_b2a)

    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(self.S, ref=np.max),
            y_axis="log",
            x_axis="time",
            ax=ax[0],
        )
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label="Onset strength")
        ax[1].vlines(
            librosa.frames_to_time(onset_raw),
            0,
            self.oenv.max(),
            label="Raw onsets",
            color="r",
        )
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label="Onset strength")
        ax[2].vlines(
            librosa.frames_to_time(onset_bt),
            0,
            self.oenv.max(),
            label="Backtracked",
            color="r",
        )
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label="RMS")
        ax[3].vlines(
            librosa.frames_to_time(onset_bt_rms),
            0,
            self.oenv.max(),
            label="Backtracked (RMS)",
            color="r",
        )
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)

    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel[0] / pose_fps + offset
        return time_vel

    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2) / (2 * sigma**2))
        dis_all_b2a /= len(b)
        return dis_all_b2a

    def calculate_align(
        self,
        onset_raw,
        onset_bt,
        onset_bt_rms,
        beat_right_arm,
        beat_right_shoulder,
        beat_right_wrist,
        beat_left_arm,
        beat_left_shoulder,
        beat_left_wrist,
        pose_fps=25,
    ):
        audio_bt = librosa.frames_to_time(onset_bt_rms)
        pose_bt = self.motion_frames2time(beat_right_wrist, 0, pose_fps)
        # avg_dis_all_b2a = self.GAHR(audio_bt, pose_bt, self.sigma)
        avg_dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
        return avg_dis_all_b2a


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n - 1)


def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


class SRGR(object):
    def __init__(self, threshold=0.1, joints=63):
        self.threshold = threshold
        self.pose_dimes = joints
        self.counter = 0
        self.sum = 0

    def run(self, results, targets, semantic):
        results = results.reshape(-1, self.pose_dimes, 3)
        targets = targets.reshape(-1, self.pose_dimes, 3)
        semantic = semantic.reshape(-1)
        diff = np.sum(abs(results - targets), 2)
        success = np.where(diff < self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1]
            success[i, :] *= semantic[i] * (1 / 0.165)
        rate = np.sum(success) / (success.shape[0] * success.shape[1])
        self.counter += success.shape[0]
        self.sum += rate * success.shape[0]
        return rate

    def avg(self):
        return self.sum / self.counter


class L1div(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0

    def run(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, 0)
        for i in range(results.shape[0]):
            results[i, :] = abs(results[i, :] - mean)
        sum_l1 = np.sum(results)
        self.sum += sum_l1

    def avg(self):
        return self.sum / self.counter


def load_fidnet_checkpoints(model, save_path, load_name="model"):
    states = torch.load(save_path)
    new_weights = OrderedDict()
    flag = False
    for k, v in states["model_state"].items():
        if "module" not in k:
            break
        else:
            new_weights[k[7:]] = v
            flag = True
    if flag:
        model.load_state_dict(new_weights)
    else:
        model.load_state_dict(states["model_state"])
    print(f"load self-pretrained checkpoints for {load_name}")


def process_motion(motion):
    # breakpoint()
    #  Put on floor
    floor_height = motion.min(axis=0).min(axis=0)[1]
    motion[:, :, 1] -= floor_height

    #  '''XZ at origin'''
    root_pos_init = motion[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    motion = motion - root_pose_init_xz

    # '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = 18, 13, 9, 5
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = (
        forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]
    )

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(motion.shape[:-1] + (4,)) * root_quat_init

    motion_b = motion.copy()

    motion = qrot_np(root_quat_init, motion)

    # all joints root relative
    motion[:, 1:, :] = motion[:, 1:, :] - motion[:, :1, :]

    # hands relative to wrist
    motion[:, 23:43, :] = motion[:, 23:43, :] - motion[:, [7], :]
    motion[:, 43:, :] = motion[:, 43:, :] - motion[:, [11], :]
    # motion[:, 23:, :] = motion[:, 23:, :] * 10
    # motion = motion * 3 # all equal scale

    # motion = motion.reshape(-1, 63 * 3)
    return motion


def smoothing(joints):
    """
    joints: Tx24x3
    """
    for i in range(5, joints.shape[0] - 5):
        # breakpoint()
        filt = joints[i - 5 : i + 5, :, :]
        # print(len(filt))
        joints[i] = np.average(filt, axis=0, weights=[0.5] * 10)
        joints[i] = filt.mean(axis=0)
    return joints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", type=str, required=True, help="directory containing the results"
    )
    args = parser.parse_args()
    result_dir = args.result_dir

    ae_path = "./experiments/eval/last_499.bin"
    FIDNet = HalfEmbeddingNet(pose_length=128, pose_dim=189, feature_length=300)
    load_fidnet_checkpoints(FIDNet, ae_path, "HalfEmbeddingNet")
    FIDNet.cuda()
    FIDNet.eval()

    alignmenter = Alignment(sigma=0.3, order=10)
    srgr_cal = SRGR(0.3, 63)
    l1_calculator = L1div()

    gt_files = glob.glob(os.path.join(result_dir, "*/*/gt.npy"))
    gt_files.sort()
    # breakpoint()

    align = 0
    counter = 0

    # gt_files = gt_files[:1000]

    for its, gt_file in tqdm(enumerate(gt_files)):
        # load the npy file
        gt = np.load(gt_file)
        pred = np.load(gt_file.replace("gt.npy", "pred.npy"))
        #
        # breakpoint()
        # TEMP
        base2_path = result_dir
        sem_path = "/".join(
            gt_file.replace("gt.npy", "sem_lsn.npy").split("/")[-3:]
        )  # this is path to semantic annotation stored in npy file.
        # reshape to posedims
        sem = np.load(os.path.join(base2_path, sem_path))
        # pred = smoothing(pred)

        gt = gt.reshape(-1, 189)
        pred = pred.reshape(-1, 189)

        # load lsn audio file and reshape (-1)
        audio_file = gt_file.replace("gt.npy", "lsn_audio.wav")
        audio, sr = librosa.load(audio_file, sr=16000)
        audio = librosa.util.normalize(audio)
        audio = audio.reshape(-1)

        # get audio pose beats and calculate alignment

        gt_np = gt.reshape(128, 63, 3).copy()
        pred_np = pred.reshape(128, 63, 3).copy()
        pred_align = pred_np.reshape(-1, 189).copy()
        gt_align = gt_np.reshape(-1, 189).copy()

        _ = srgr_cal.run(pred, gt, sem)

        _ = l1_calculator.run(pred.copy())

        gt_np = process_motion(gt_np)
        pred_np = process_motion(pred_np)

        if its == 0:
            pred_all = pred_np[np.newaxis, :]
            tar_all = gt_np[np.newaxis, :]
        else:
            pred_all = np.concatenate([pred_all, pred_np[np.newaxis, :]], axis=0)
            tar_all = np.concatenate([tar_all, gt_np[np.newaxis, :]], axis=0)

        # get audio pose beats and calculate alignment
        # onset_raw, onset_bt, onset_bt_rms = alignmenter.load_audio(audio, 0, 128/25, True)
        onset_raw, onset_bt, onset_bt_rms = alignmenter.load_audio(
            audio, 0, 128 / 25, True
        )
        if onset_raw is None:
            print("skipping", gt_file)
            # breakpoint()
            continue
        else:
            counter += 1
            (
                beat_right_arm,
                beat_right_shoulder,
                beat_right_wrist,
                beat_left_arm,
                beat_left_shoulder,
                beat_left_wrist,
            ) = alignmenter.load_pose(pred_align, 0, 128 / 25, 25, True)
            align += alignmenter.calculate_align(
                onset_raw,
                onset_bt,
                onset_bt_rms,
                beat_right_arm,
                beat_right_shoulder,
                beat_right_wrist,
                beat_left_arm,
                beat_left_shoulder,
                beat_left_wrist,
                25,
            )

    print("Alignment:", align / counter)
    print(counter)

    print(f"div pred {calculate_avg_distance(pred_all)}")
    print(f"div tar {calculate_avg_distance(tar_all)}")

    l1div = l1_calculator.avg()
    print(f"l1div score: {l1div}")

    srgr = srgr_cal.avg()
    print(f"srgr score: {srgr}")
