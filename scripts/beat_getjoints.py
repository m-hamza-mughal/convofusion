import os
import numpy as np

from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *


joint_list = [
    'HeadEnd',
    'Neck1',
    'LeftShoulder',
    'Spine',
    'Spine1',
    'LeftArm',
    'LeftForeArm',
    'LeftHand',
    'LeftHandMid',
    'RightArm',
    'RightForeArm',
    'RightHand',
    'RightHandMid',
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'LeftToeBase',
    'LeftToeBaseEnd',
    'RightUpLeg',
    'RightLeg',
    'RightFoot',
    'RightToeBase',
    'RightToeBaseEnd',

    'LeftHand',
    'LeftHandThumb1',
    'LeftHandThumb2',
    'LeftHandThumb3',
    'LeftHandThumb4',
    'LeftHandIndex1',
    'LeftHandIndex2',
    'LeftHandIndex3',
    'LeftHandIndex4',
    'LeftHandMiddle1',
    'LeftHandMiddle2',
    'LeftHandMiddle3',
    'LeftHandMiddle4',
    'LeftHandRing1',
    'LeftHandRing2',
    'LeftHandRing3',
    'LeftHandRing4',
    'LeftHandPinky1',
    'LeftHandPinky2',
    'LeftHandPinky3',
    'LeftHandPinky4',
    'LeftHandMid',

    'RightHand',
    'RightHandThumb1',
    'RightHandThumb2',
    'RightHandThumb3',
    'RightHandThumb4',
    'RightHandIndex1',
    'RightHandIndex2',
    'RightHandIndex3',
    'RightHandIndex4',
    'RightHandMiddle1',
    'RightHandMiddle2',
    'RightHandMiddle3',
    'RightHandMiddle4',
    'RightHandRing1',
    'RightHandRing2',
    'RightHandRing3',
    'RightHandRing4',
    'RightHandPinky1',
    'RightHandPinky2',
    'RightHandPinky3',
    'RightHandPinky4',
    'RightHandMid'
]


def convert_beatskl_to_dndskel(i, path, outpath):
    """
    Convert BEAT skeleton files to DND skeleton files

    Parameters
    ----------
    i : int
        Speaker ID
    path : str
        Path to BEAT dataset
    outpath : str
        Path to save DND skeleton format files for BEAT motions
    """
    for bvh in os.listdir(path+str(i+1)+"/"):
        if bvh.endswith('.bvh'):
            if os.path.exists(path+str(i+1)+"/"+bvh[:-3]+"npy"):
                continue
            p = BVHParser()
            filename = os.path.join(path, str(i+1), bvh)
            print(filename)

            try:
                data = [p.parse(filename)]
            except Exception as e:
                print("Error in file: ", filename, e)
                continue

            dr_pipe = Pipeline([
                ('param', MocapParameterizer('position')),
            ])

            xx = dr_pipe.fit_transform(data)
            df = xx[0].values.head(-1)

            data_list = []

            for f in range(data[0].values.shape[0]-1):
                p_in_f = []
                for joint in joint_list:
                    if joint == 'LeftHandMid':
                        x = (df['LeftHand_Xposition'][f] + df['LeftHandIndex1_Xposition'][f] + df['LeftHandRing1_Xposition']
                             [f] + df['LeftHandPinky1_Xposition'][f] + df['LeftHandThumb1_Xposition'][f])/5
                        y = (df['LeftHand_Yposition'][f] + df['LeftHandIndex1_Yposition'][f] + df['LeftHandRing1_Yposition']
                             [f] + df['LeftHandPinky1_Yposition'][f] + df['LeftHandThumb1_Yposition'][f])/5
                        z = (df['LeftHand_Zposition'][f] + df['LeftHandIndex1_Zposition'][f] + df['LeftHandRing1_Zposition']
                             [f] + df['LeftHandPinky1_Zposition'][f] + df['LeftHandThumb1_Zposition'][f])/5
                    elif joint == 'RightHandMid':
                        x = (df['RightHand_Xposition'][f] + df['RightHandIndex1_Xposition'][f] + df['RightHandRing1_Xposition']
                             [f] + df['RightHandPinky1_Xposition'][f] + df['RightHandThumb1_Xposition'][f])/5
                        y = (df['RightHand_Yposition'][f] + df['RightHandIndex1_Yposition'][f] + df['RightHandRing1_Yposition']
                             [f] + df['RightHandPinky1_Yposition'][f] + df['RightHandThumb1_Yposition'][f])/5
                        z = (df['RightHand_Zposition'][f] + df['RightHandIndex1_Zposition'][f] + df['RightHandRing1_Zposition']
                             [f] + df['RightHandPinky1_Zposition'][f] + df['RightHandThumb1_Zposition'][f])/5
                    else:
                        x = df['%s_Xposition' % joint][f]
                        y = df['%s_Yposition' % joint][f]
                        z = df['%s_Zposition' % joint][f]

                    p = [x, y, z]
                    p_in_f.append(p)
                #
                data_list.append(p_in_f)
            # print(np.array(data_list).shape)
            if not os.path.exists(os.path.join(outpath, str(i+1))):
                os.makedirs(os.path.join(outpath, str(i+1)))
            np.save(os.path.join(outpath, str(i+1),
                    bvh[:-3] + 'npy'), np.array(data_list))

    return i+1


if __name__ == '__main__':
    beat_path = './datasets/beat_english_v0.2.1/' # CHANGE AS NEEDED
    out_folder = beat_path # change to a different folder if needed

    for s in range(1, 31):  # 30 speakers
        convert_beatskl_to_dndskel(s, beat_path, out_folder)
