import os
import glob
import argparse
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import ListedColormap
import shutil
from distutils.dir_util import copy_tree

import subprocess
import shlex
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
from time import time
import random

matplotlib.use("Agg")


def plot_3d_motion(
    save_path,
    kinematic_tree,
    joints,
    title,
    text="",
    figsize=(10, 10),
    fps=120,
    radius=4,
    color_range=None,
):
    #     matplotlib.use('Agg')

    title_sp = title.split(" ")
    divider = 10
    if len(title_sp) > divider:
        chunks = np.ceil(len(title_sp) / divider).astype(int)
        title = "\n".join(
            [" ".join(title_sp[i * divider : (i + 1) * divider]) for i in range(chunks)]
        )

    text_sp = text.split(" ")
    if len(text_sp) > divider:
        chunks = np.ceil(len(text_sp) / divider).astype(int)
        text = "\n".join(
            [" ".join(text_sp[i * divider : (i + 1) * divider]) for i in range(chunks)]
        )

    title = title + "\n" + text

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=15)
        ax.grid(b=False)
        ax.view_init(elev=135, azim=-90)
        ax.set_proj_type("persp")

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]

        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.2))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)

    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    # if colors is None:
    colors = [
        "red",
        "blue",
        "black",
        "red",
        "blue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
    ]
    frame_number = data.shape[0]
    #     print(data.shape)

    data[:, :, 2] += 0.35
    data *= 1.5

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset

    trajec = data[:, 0, [0, 2]]

    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)

        ax.lines = []
        ax.collections = []

        ax.dist = 4
        
        if index > 1:
            ax.plot3D(
                trajec[:index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1],
                linewidth=1.0,
                color="blue",
            )
            # ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        colors = [
            "red",
            "blue",
            "black",
            "red",
            "blue",
            "darkblue",
            "darkblue",
            "darkblue",
            "darkblue",
            "darkblue",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
        ]
        if color_range:
            if index in range(color_range[0], color_range[1]):
                colors = ["black"] * len(colors)
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):

            if i < 5:
                linewidth = 4.0  # *10
                color = [41 / 255, 81 / 255, 209 / 255, 0.7]
                for i, point in enumerate(chain):
                    psize = 50.0
                    ax.scatter(
                        data[index, point, 0],
                        data[index, point, 1],
                        data[index, point, 2],
                        color="#382be3",
                        s=psize,
                    )
            else:
                linewidth = 2.0  # *10
                color = [254 / 255, 78 / 255, 126 / 255, 0.9]
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
                zorder=1.1,
            )
            # print(index, chain)

            # print(trajec[:index, 0].shape)

        plot_xzPlane(
            MINS[0] - trajec[0, 0] - 3,
            MAXS[0] - trajec[0, 0] + 3,
            0,
            MINS[2] - trajec[0, 1],
            MAXS[2] - trajec[0, 1] + 3,
        )

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # # plt.savefig(os.path.dirname(save_path) + '/figs3/%05d.svg' % index, bbox_inches='tight', pad_inches=0)
        # if index > 2:
        #  raise

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    ani.save(save_path, fps=fps)
    plt.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--src_dir", type=str, required=True)
    argparser.add_argument("--without_WEG_dir", type=str, default=None)
    args = argparser.parse_args()
    src_dir = args.src_dir
    wosemantic_dir = args.without_WEG_dir

    beatdnd_kinematic_chain = [
        [0, 4, 3, 2, 1],  # spine
        [0, 18, 19, 20, 21, 22],  # right leg
        [0, 13, 14, 15, 16, 17],  # left leg
        [3, 9, 10, 11],  # right arm
        [3, 5, 6, 7],  # left arm
        [7, 23, 24, 25, 26],  # left thumb
        [7, 27, 28, 29, 30],  # left index
        [7, 8, 31, 32, 33, 34],  # left middle
        [7, 35, 36, 37, 38],  # left ring
        [7, 39, 40, 41, 42],  # left pinky
        [11, 43, 44, 45, 46],  # right thumb
        [11, 47, 48, 49, 50],  # right index
        [11, 12, 51, 52, 53, 54],  # right middle
        [11, 55, 56, 57, 58],  # right ring
        [11, 59, 60, 61, 62],  # right pinky
    ]

    random.seed(0)
    set_paths_orig = glob.glob(src_dir + "*/*/*")
    set_paths = set_paths_orig

    #
    set_paths = random.sample(set_paths, len(set_paths))
    # set_paths = + set_paths
    print(str(len(set_paths)))
    print(set_paths[0])

    # breakpoint()

    for reaction_set in set_paths:
        name = "/".join(reaction_set.split("/")[-3:])
        try:
            lsn_gt = np.load(pjoin(reaction_set, "gt.npy"))
            lsn_pred = np.load(pjoin(reaction_set, "pred.npy"))

            if wosemantic_dir is not None:
                sem_path = os.path.join(wosemantic_dir, name)
            else:
                sem_path = ""

            fword_path = pjoin(reaction_set, "focus_words_lsn.txt")
            with open(fword_path, "r") as f:
                # focus_words = f.read().split(',')
                focus_words = f.readlines()

            # if focus_words empty
            if len(focus_words) == 0:
                continue

            focus_words = ["[" + x.strip() + "]" for x in focus_words]
            print(focus_words)

        except FileNotFoundError as e:
            print(e)
            continue

        if os.path.exists(sem_path):
            lsn_semantic = np.load(pjoin(sem_path, "pred.npy"))
        # load text data
        save_path_pred = pjoin(reaction_set, "pred_lsn.mp4")
        save_path_gt = pjoin(reaction_set, "gt_lsn.mp4")
        save_path_sem = pjoin(reaction_set, "pred_sem.mp4")

        save_path_pred_audio = pjoin(reaction_set, "pred_audio.mp4")
        save_path_gt_audio = pjoin(reaction_set, "gt_audio.mp4")
        save_path_sem_audio = pjoin(reaction_set, "sem_audio.mp4")

        print(name)

        #
        # plot pred
        plot_3d_motion(
            save_path_pred,
            beatdnd_kinematic_chain,
            lsn_pred,
            title="",
            text="WEG on:" + ",".join(focus_words),
            fps=25,
            radius=4,
        )
        # plot_3d_motion(save_path_pred.replace('pred', 'pred2'), beatdnd_kinematic_chain, lsn_pred, title="", text='', fps=25, radius=4)

        # # plot gt
        plot_3d_motion(
            save_path_gt,
            beatdnd_kinematic_chain,
            lsn_gt,
            title="",
            text="",
            fps=25,
            radius=4,
        )

        # # plot sem
        if os.path.exists(sem_path):
            plot_3d_motion(
                save_path_sem,
                beatdnd_kinematic_chain,
                lsn_semantic,
                title="",
                text="No WEG off:" + ",".join(focus_words),
                fps=25,
                radius=4,
            )

        # # ffmpeg add audio to video
        subprocess.run(
            shlex.split(
                f"ffmpeg -i {save_path_pred} -i {pjoin(reaction_set, 'lsn_audio.wav')} -map 0:v -map 1:a -c:v copy -acodec mp3 {save_path_pred_audio} -y -loglevel error"
            )
        )
        subprocess.run(
            shlex.split(
                f"ffmpeg -i {save_path_gt} -i {pjoin(reaction_set, 'lsn_audio.wav')} -map 0:v -map 1:a -c:v copy -acodec mp3 {save_path_gt_audio} -y -loglevel error"
            )
        )
        #
        if os.path.exists(sem_path):
            subprocess.run(
                shlex.split(
                    f"ffmpeg -i {save_path_sem} -i {pjoin(reaction_set, 'lsn_audio.wav')} -map 0:v -map 1:a -c:v copy -acodec mp3 {save_path_sem_audio} -y -loglevel error"
                )
            )

        if os.path.exists(sem_path):
            save_path_combined = pjoin(reaction_set, "combined.mp4")
            save_path_combined_audio = pjoin(reaction_set, "combined_audio.mp4")
            subprocess.run(
                shlex.split(
                    f'ffmpeg -i {save_path_gt_audio} -i {save_path_pred_audio} -i {save_path_sem_audio} -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]"  -acodec mp3 {save_path_combined} -y -loglevel error'
                )
            )
            subprocess.run(
                shlex.split(
                    f"ffmpeg -i {save_path_combined} -i {pjoin(reaction_set, 'lsn_audio.wav')} -map 0:v -map 1:a -c:v copy -acodec mp3 {save_path_combined_audio} -y -loglevel error"
                )
            )
