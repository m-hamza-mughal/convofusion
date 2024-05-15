"""
combine hand and body csvs for each session
make session folders with audios and joint csvs

--- 
read session
    - read all motion csvs
    - read all audios 
    - extract persons from id of motion csvs

    for each person
    - read csv
    -- reshape feats
    -- add reorder joints
    -- 
    - read audio (resample)
    - divide based on utterance and get timestamps
    - for each utterance based on audio/timestamps (while loop until count == length of utterance start from count 0)
    -- discard if length is less than 3.5 second and continue
    -- split if length is more than 5 second and add to utterances and continue
    -- make folder based on sessid and personid and utteranceid
    -- get motion_spk start from audio start and end at start + max len
    -- get motion_lsn1, motion_lsn2, motion_lsn3,  motion_lsn4

    -- take audio_spk start from audio start and end at start + max len
    -- take audio_lsn1, audio_lsn2, audio_lsn3, audio_lsn4

    -- transcribe audios to text
    -- increase count
"""

import os
import sys
import glob
import json
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import librosa
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, detect_silence
from scipy.interpolate import interp1d
import whisper
import itertools
import logging

# from sklearn.preprocessing import MinMaxScaler

# configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# np.random.seed(0)


def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    """
    Returns list of audio segments from splitting audio_segment on silent sections
    audio_segment - original pydub.AudioSegment() object
    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms
    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS
    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms
    seek_step - step size for interating over the segment in ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start,end)
            in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return [
        audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
        for start,end in output_ranges
    ], output_ranges


def transcribe(audio):
    model = whisper.load_model("medium.en")
    audio_array = audio.get_array_of_samples()
    # 
    audio_np = np.array(audio_array).T.astype(np.float32)
    audio_np /= np.iinfo(audio_array.typecode).max 
    result = model.transcribe(audio_np)
    
    segments = [(x["start"], x["end"], x["text"]) for x in result["segments"]]
    return result["text"], segments

def process_session(session_path, output_folder):
    logging.info(os.path.basename(session_path))

    start_motion_frame = {
            "joints_24_10_22": 185,
            "joints_28_10_22": 166,
            "joints_03_02_23": 88,
            "joints_03_02_23_nonverbalremoved": 88,
            "joints_24_02_23": 0,
        }

    max_audio_len = int(((128 * 6)/25) * 1000) # 128 frames at 25 fps
    max_motion_len = 128 * 6# 128 frames at 25 fps

    # make session output folder
    session_name = os.path.basename(session_path)
    session_output_folder = os.path.join(output_folder, session_name)
    os.makedirs(session_output_folder, exist_ok=True)
    
    # read all motion csvs in subfolders
    person_paths = glob.glob(os.path.join(session_path, '*'))
    
    motion_dfs = {}
    audio_dfs = {}
    person_names = []
    for p_path in person_paths:
        a_file = glob.glob(os.path.join(p_path, '*.wav'))[0]
        m_csv = glob.glob(os.path.join(p_path, '*.csv'))[0]
        name = os.path.basename(p_path)
        csv_df = pd.read_csv(m_csv, header=None)
        frame_array = csv_df.values[:, 1:].reshape(-1, 67, 3)
        frame_array = frame_array[:, [3] + list(range(0,3)) + list(range(4,frame_array.shape[1])) , :]
        start_idx = start_motion_frame[os.path.basename(session_path)]
        frame_array = frame_array[start_idx:]

        # Resample motion to 25 fps for joints_28_10_22
        if os.path.basename(session_path) == "joints_28_10_22":
            xp = np.arange(0, len(frame_array), 30/25)

            if xp[-1] > len(frame_array)-1:
                xp = xp[:-1]
            
            f = interp1d(np.arange(len(frame_array)), frame_array, axis=0)
            motion = f(xp)

            motion_dfs[name] = motion
        else:
            motion_dfs[name] = frame_array

        audio_seg = AudioSegment.from_wav(a_file)
        audio_seg = audio_seg.set_frame_rate(16000)
        # audio_seg = audio_seg.normalize()
        audio_dfs[name] = audio_seg

        person_names.append(name)
    
    person_names = sorted(person_names)
    # person_names = person_names[2:] # remove the first two people for debugging ben
    for name in person_names:
        # if name in ["anne", "ben"]:
        #     continue
        logging.info(name)
        # breakpoint()
        audio_person = audio_dfs[name]
        motion_person = motion_dfs[name]
        # get utterances and their timestamps
        segments, output_ranges = split_on_silence(audio_segment=audio_person, min_silence_len=1000, silence_thresh=-45, keep_silence=10)
        print(len(segments))
        seg_count = 0
        # prev_start = -max_audio_len
        while seg_count < len(segments):
            # breakpoint()
            seg = segments[seg_count]
            start, end = output_ranges[seg_count]
            start = 0 if start < 0 else start
            seg_count += 1
            if seg_count != 1 and start < prev_start + max_audio_len:
                if end > prev_start + max_audio_len:
                    start = prev_start + max_audio_len
                else:
                    continue
            
            # print(len(seg), start, end)
            # # discard seg if it silence is more than 20% of the segment
            # if len(seg) < int(max_audio_len * 0.8):
            #     continue

            # # if seg is longer than 5 seconds, split it into segments less than 5 seconds
            if len(seg) > max_audio_len:
                # print("len(seg) > max_audio_len", len(seg), start, end)
                seg_remain = audio_person[start + max_audio_len:end] # cant be seg[max_audio_len:end] because start of seg and start now are different
                # print("seg_remain", len(seg_remain))
                # seg = seg[:start + max_ausdio_len]
                segments.insert(seg_count, seg_remain)
                output_ranges.insert(seg_count, [start + max_audio_len, end])
                # breakpoint()
                # sub_segs, sub_ranges = split_on_silence(audio_segment=audio_person[start:end], min_silence_len=300, silence_thresh=-45, keep_silence=10)
                # print(sub_ranges)
                # sub_ranges = [[start + r[0], start + r[1]] for r in sub_ranges]
                # print(sub_ranges)
                # segments[seg_count:seg_count] = sub_segs
                # output_ranges[seg_count:seg_count] = sub_ranges
                # continue

            # assert len(seg) >= int(max_audio_len * 0.8) and len(seg) <= max_audio_len
            orig_start = start
            # breakpoint()
            start_list = [start] 
            # to augment the data by adding random start times around original start time
            # start_list = start_list + np.random.randint(max(0, start - 2000), start + 2000, size=4).tolist()

            for start in start_list:
                utterance_id = f"{name}_{start}_{start + max_audio_len}"
                start = max(0, start - 1000)
                # max_motion_len = max_motion_len + 50
                # max_audio_len = int(((max_motion_len)/25) * 1000)


                m_start = int((start / 1000) * 25) # divided by 1000 because start is in milliseconds
                # get motion and audio for this utterance
                motion_spk = motion_person[m_start:m_start + max_motion_len]
                audio_spk = audio_person[start:start + max_audio_len]
                

                # get motion and audio for other speakers
                other_persons = [p for p in person_names if p != name]
                assert len(other_persons) == 4
                motion_lsn1 = motion_dfs[other_persons[0]][m_start:m_start + max_motion_len]
                motion_lsn2 = motion_dfs[other_persons[1]][m_start:m_start + max_motion_len]
                motion_lsn3 = motion_dfs[other_persons[2]][m_start:m_start + max_motion_len]
                motion_lsn4 = motion_dfs[other_persons[3]][m_start:m_start + max_motion_len]

                audio_lsn1 = audio_dfs[other_persons[0]][start:start + max_audio_len]
                
                audio_lsn2 = audio_dfs[other_persons[1]][start:start + max_audio_len]
                
                audio_lsn3 = audio_dfs[other_persons[2]][start:start + max_audio_len]
                
                audio_lsn4 = audio_dfs[other_persons[3]][start:start + max_audio_len]
                

                # make folder for this utterance
                utterance_folder = os.path.join(session_output_folder, utterance_id)
                os.makedirs(utterance_folder, exist_ok=True)

                # transcribe audio to text
                if len(detect_silence(audio_spk, silence_thresh=-40, min_silence_len=200)) > 1:
                    text_spk, seg_spk = transcribe(audio_spk)
                    text_spk = "" if text_spk is None else text_spk
                else:
                    text_spk = ""
                    seg_spk = []
                
                if len(detect_silence(audio_lsn1, silence_thresh=-40, min_silence_len=200)) > 1:
                    text_lsn1, seg_lsn1 = transcribe(audio_lsn1)
                    text_lsn1 = "" if text_lsn1 is None else text_lsn1
                else:
                    text_lsn1 = ""
                    seg_lsn1 = []
                
                if len(detect_silence(audio_lsn2, silence_thresh=-40, min_silence_len=200)) > 1:
                    text_lsn2, seg_lsn2 = transcribe(audio_lsn2)
                    text_lsn2 = "" if text_lsn2 is None else text_lsn2
                else:
                    text_lsn2 = ""
                    seg_lsn2 = []
                
                if len(detect_silence(audio_lsn3, silence_thresh=-40, min_silence_len=200)) > 1:
                    text_lsn3, seg_lsn3 = transcribe(audio_lsn3)
                    text_lsn3 = "" if text_lsn3 is None else text_lsn3
                else:
                    text_lsn3 = ""
                    seg_lsn3 = []
                
                if len(detect_silence(audio_lsn4, silence_thresh=-40, min_silence_len=200)) > 1:
                    text_lsn4, seg_lsn4 = transcribe(audio_lsn4)
                    text_lsn4 = "" if text_lsn4 is None else text_lsn4
                else:
                    text_lsn4 = ""
                    seg_lsn4 = []
                
                

                # save motion and audio
                np.save(os.path.join(utterance_folder, 'motion_spk.npy'), motion_spk)
                np.save(os.path.join(utterance_folder, 'motion_lsn1.npy'), motion_lsn1)
                np.save(os.path.join(utterance_folder, 'motion_lsn2.npy'), motion_lsn2)
                np.save(os.path.join(utterance_folder, 'motion_lsn3.npy'), motion_lsn3)
                np.save(os.path.join(utterance_folder, 'motion_lsn4.npy'), motion_lsn4)

                audio_spk.export(os.path.join(utterance_folder, 'audio_spk.wav'), format='wav')
                audio_lsn1.export(os.path.join(utterance_folder, 'audio_lsn1.wav'), format='wav')
                audio_lsn2.export(os.path.join(utterance_folder, 'audio_lsn2.wav'), format='wav')
                audio_lsn3.export(os.path.join(utterance_folder, 'audio_lsn3.wav'), format='wav')
                audio_lsn4.export(os.path.join(utterance_folder, 'audio_lsn4.wav'), format='wav')

                logging.info("".join(["spk ", text_spk, "lsn1 ", text_lsn1, "lsn2 ", text_lsn2, "lsn3 ", text_lsn3, "lsn4 ", text_lsn4]))
                # save text
                with open(os.path.join(utterance_folder, 'text_spk.txt'), 'w') as f:
                    f.write(text_spk)
                with open(os.path.join(utterance_folder, 'text_lsn1.txt'), 'w') as f:
                    f.write(text_lsn1)
                with open(os.path.join(utterance_folder, 'text_lsn2.txt'), 'w') as f:
                    f.write(text_lsn2)
                with open(os.path.join(utterance_folder, 'text_lsn3.txt'), 'w') as f:
                    f.write(text_lsn3)
                with open(os.path.join(utterance_folder, 'text_lsn4.txt'), 'w') as f:
                    f.write(text_lsn4)

                if max_motion_len > 128:
                    with open(os.path.join(utterance_folder, 'seg_spk.txt'), 'w') as f:
                        seg_lines = [str(start) + ',' + str(end) + '---' + txt + '\n' for start, end, txt in seg_spk]
                        f.writelines(seg_lines)
                    with open(os.path.join(utterance_folder, 'seg_lsn1.txt'), 'w') as f:
                        seg_lines = [str(start) + ',' + str(end) + '---' + txt + '\n' for start, end, txt in seg_lsn1]
                        f.writelines(seg_lines)
                    with open(os.path.join(utterance_folder, 'seg_lsn2.txt'), 'w') as f:
                        seg_lines = [str(start) + ',' + str(end) + '---' + txt + '\n' for start, end, txt in seg_lsn2]
                        f.writelines(seg_lines)
                    with open(os.path.join(utterance_folder, 'seg_lsn3.txt'), 'w') as f:
                        seg_lines = [str(start) + ',' + str(end) + '---' + txt + '\n' for start, end, txt in seg_lsn3]
                        f.writelines(seg_lines)
                    with open(os.path.join(utterance_folder, 'seg_lsn4.txt'), 'w') as f:
                        seg_lines = [str(start) + ',' + str(end) + '---' + txt + '\n' for start, end, txt in seg_lsn4]
                        f.writelines(seg_lines)
                
            # breakpoint()
            prev_start = orig_start
            # prev_end = end
            

def process_utterance(session_path, person_id, utterance_id, audio_path, motion_path):
    pass



if __name__ == "__main__":
    logging.info(msg="Starting to process sessions")
    session_parent_folder = '/CT/GroupGesture/work/DnD_first_three_recordings/joint_pos/joint_csv/final/'
    session_folders = glob.glob(os.path.join(session_parent_folder, '*'))
    session_folders = ['/CT/GroupGesture/work/DnD_first_three_recordings/joint_pos/joint_csv/final/joints_28_10_22']
    for session_folder in session_folders:
        process_session(session_path=session_folder, output_folder='/CT/GroupGesture/work/GestureSynth/ut_data_30sec')
