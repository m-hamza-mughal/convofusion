import codecs as cs
import os
import random
from os.path import join as pjoin

import numpy as np
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import glob
from scipy.interpolate import interp1d
import librosa
import pandas as pd

from .utils.text_utils import parse_textgrid
from .utils.motion_rep_utils import convert_euler_to_6D, forward_kinematics_cont6d, forward_kinematics_euler
from .utils.quaternion import qbetween_np, qrot_np
import time
import soundfile as sf

#
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

def collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[1], reverse=True)

    # 
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
        [b[9] for b in notnone_batches],
        "name":
        [b[10] for b in notnone_batches],
        "spk_name": [b[11] for b in notnone_batches],
        "lsn_name": [b[12] for b in notnone_batches],
        "lsn_id": [b[13] for b in notnone_batches],
    }
    return adapted_batch



"""Dataset class for BEAT and DnD dataset combined for monadic/dyadic generation task"""
class BEATAugReactionDataset(data.Dataset):
    def __init__(
        self,
        split_file,
        max_motion_length,
        min_motion_length,
        motion_rep,
        unit_length,
        dataset_path,
        debug=False,
        tiny=False,
        **kwargs,
    ):
        # 
        self.motion_rep = motion_rep
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length # this is used to convert motion length into multiples of unit_length
        self.njoints = 63
        self.face_joint_idx = kwargs['face_joint_idx']
        self.SR = kwargs['sample_rate']
        self.N_MELS = kwargs['num_mels']
        self.HOP_LEN = kwargs['hop_length']
        self.FPS = kwargs['fps']
        self.dataset_select = kwargs['dataset_select']
        # breakpoint()
        self.dnd_speaker_names = ['anne', 'ben', 'chris', 'jack', 'lilas']
        self.beat_speaker_names = ['wayne', 'scott', 'solomon', 'lawrence', 'stewart', 'carla', 'sophie', 'catherine', 'miranda', 'kieks', 'nidal', 'zhao', 'lu', 'zhang', 'carlos', 'jorge', 'itoi', 'daiki', 'jaime', 'li', 'ayana', 'luqi', 'hailing', 'kexin', 'goto', 'reamey', 'yingqing', 'tiffnay', 'hanieh', 'katya']
        self.speaker_names = self.dnd_speaker_names + self.beat_speaker_names

        data_dict = {}
        name_list = []
        length_list = []
        # breakpoint()

        self.beat_split_file = split_file[0]
        self.dnd_split_file = split_file[1]

        beat_dataset_path = dataset_path[0]
        dnd_dataset_path = dataset_path[1]

        # load motion from BEAT dataset

        self.beat_split = np.loadtxt(self.beat_split_file, dtype=str)
        assert self.motion_rep == 'pos', 'motion_rep should be pos for BEAT dataset'
        self.beat_motion_paths = [path for path in glob.glob(os.path.join(beat_dataset_path, '*/*.npy')) if 'euler' not in path]
        
        if debug:
            self.beat_split = self.beat_split[:10]
        if tiny:
            self.beat_split = self.beat_split[:5]
        # self.beat_split = self.beat_split[:10]
        self.beat_motion_paths.sort()
        if self.dataset_select == 'dnd':
            self.beat_motion_paths = []

        beat_names = []

        print("Loading BEAT dataset from {}".format(beat_dataset_path))
        print(self.beat_split_file)
        for motion_path in tqdm(self.beat_motion_paths):
            motion_name = os.path.basename(motion_path).replace('.npy', '')
            if motion_name not in self.beat_split:
                continue

            # continue
            
            orig_motion = np.load(motion_path)
            text_path = motion_path.replace('.npy', '.TextGrid').replace('_euler', '')
            audio_path = motion_path.replace('.npy', '.wav').replace('_euler', '')
            sem_path = motion_path.replace('.npy', '.txt').replace('_euler', '')

            # rescale frame rate to 25 frames per second to match with DnD dataset
            xp = np.arange(0, len(orig_motion), 120/25)

            if xp[-1] > len(orig_motion)-1:
                xp = xp[:-1]
            
            f = interp1d(np.arange(len(orig_motion)), orig_motion, axis=0)
            motion = f(xp) # (motion length, 228)
            

            if motion.shape[0] < self.max_motion_length:
                raise

            assert self.motion_rep == 'pos', 'motion_rep should be pos for BEAT dataset'
            # breakpoint()
            # reorder joints to make root joint first
            motion = motion[:, [3] + list(range(0,3)) + list(range(4, motion.shape[1])) , :] 
            motion = motion * 10 # cm to mm to match with dnd dataset
            motion = motion[:motion.shape[0] - motion.shape[0] % self.max_motion_length] # (n_chunks * window_size, dim)
            motion_chunks = np.array_split(motion, motion.shape[0] // self.max_motion_length, axis=0) # (n_chunks, window_size, dim)
            
            # breakpoint()
            for idx, chunk in enumerate(motion_chunks):
                start_idx = idx * self.max_motion_length
                motion_lsn = chunk

                motion_lsn = self.process_motion([motion_lsn])[0]

                text_lsn, seg_lsn = self.beat_extract_text(text_path, start_idx, self.max_motion_length)
                # 
                audio_lsn = self.beat_extract_audio(audio_path, start_idx, self.max_motion_length)
                sem_lsn, sem_info = self.beat_extract_sem(sem_path, start_idx, self.max_motion_length)
                # breakpoint()

                # breakpoint()
                active_passive_bit = self.check_audio(audio_lsn) # change for beat apb
                # active_passive_bit = np.array([1]* (self.max_motion_length // 16)) # because everything is active in beat dataset

                melspec_lsn = self.get_melspecs([audio_lsn])[0]

                uncond_mel = -90 * np.ones_like(melspec_lsn)
                uncond_mel[..., 40:45] = 0
                set_name = motion_name + '/' + str(idx)
                data_dict['beat+' + set_name] = {
                    'motion_spk': np.zeros_like(motion_lsn),   
                    'motions_lsn': [motion_lsn],
                    'melspec_spk': uncond_mel, #melspec_spk,
                    'melspecs_lsn': [melspec_lsn],
                    'text_spk': "-"*10,
                    'texts_lsn': [text_lsn],
                    'audio_spk': np.zeros_like(audio_lsn),
                    'audios_lsn': [audio_lsn],
                    'active_passive_bit': [active_passive_bit],
                    'seg_lsn': seg_lsn,
                    'seg_spk': "-"*10,
                    'sem_lsn': sem_lsn,
                    'sem_info': sem_info,
                }
                name_list.append('beat+' + set_name)
                beat_names.append('beat+' + set_name)
                length_list.append(motion_lsn.shape[0])
            
        # breakpoint()

        
        print("Length of loaded data from BEAT: %d" % len(beat_names))
        

        # load motion from DnD dataset
        self.dnd_dataset_path = dnd_dataset_path
        print(self.dnd_split_file)
        self.dnd_split = np.loadtxt(self.dnd_split_file, dtype=str)
        assert self.motion_rep == 'pos', 'motion_rep should be pos for BEAT dataset'
        self.reaction_set_paths = glob.glob(os.path.join(dnd_dataset_path, '*/*'))
        
        # self.dnd_split = self.dnd_split[:500]
        if debug:
            self.dnd_split = self.dnd_split[:10]
        if tiny:
            self.dnd_split = self.dnd_split[:5] 
        self.dnd_split = self.dnd_split[:10]
        
        self.reaction_set_paths.sort()
        if self.dataset_select == 'beat':
            self.reaction_set_paths = []
        # breakpoint()
        # self.reaction_set_paths = [os.path.join(dataset_path, "joints_03_02_23/chris_2195944_2201064")]*32
        
        dnd_names = []
        # id_list = []
        print("Loading DnD dataset from {}".format(self.dnd_dataset_path))
        for set_path in tqdm(self.reaction_set_paths):
            set_name = "/".join(set_path.split('/')[-2:])

            if set_name not in self.dnd_split:
                continue

            # if 'chris' not in set_name:
            #     continue
            
            # breakpoint()
            # load motion
            try:
                motion_spk = np.load(os.path.join(set_path, 'motion_spk.npy'))
                if motion_spk.shape[0] != self.max_motion_length:
                    # print(motion_spk.shape[0], set_path)
                    continue

                motion_lsn1 = np.load(os.path.join(set_path, 'motion_lsn1.npy'))
                motion_lsn2 = np.load(os.path.join(set_path, 'motion_lsn2.npy'))
                motion_lsn3 = np.load(os.path.join(set_path, 'motion_lsn3.npy'))
                motion_lsn4 = np.load(os.path.join(set_path, 'motion_lsn4.npy'))
            except FileNotFoundError:
                continue

            motion_spk, motion_lsn1, motion_lsn2, motion_lsn3, motion_lsn4 = \
                self.process_motion([motion_spk, motion_lsn1, motion_lsn2, motion_lsn3, motion_lsn4])

            # load audio
            audio_spk = librosa.load(os.path.join(set_path, 'audio_spk.wav'), sr=self.SR)[0]
            audio_lsn1 = librosa.load(os.path.join(set_path, 'audio_lsn1.wav'), sr=self.SR)[0]
            audio_lsn2 = librosa.load(os.path.join(set_path, 'audio_lsn2.wav'), sr=self.SR)[0]
            audio_lsn3 = librosa.load(os.path.join(set_path, 'audio_lsn3.wav'), sr=self.SR)[0]
            audio_lsn4 = librosa.load(os.path.join(set_path, 'audio_lsn4.wav'), sr=self.SR)[0]

            if len(audio_spk) < (self.max_motion_length / self.FPS)*self.SR:
                print(len(audio_spk), (self.max_motion_length / self.FPS)*self.SR, set_path)
                continue
            # print("asdasd")
            audio_lsn1 = np.zeros_like(audio_spk) if len(audio_lsn1) == 0 else audio_lsn1
            audio_lsn2 = np.zeros_like(audio_spk) if len(audio_lsn2) == 0 else audio_lsn2
            audio_lsn3 = np.zeros_like(audio_spk) if len(audio_lsn3) == 0 else audio_lsn3
            audio_lsn4 = np.zeros_like(audio_spk) if len(audio_lsn4) == 0 else audio_lsn4

            # try:
            # breakpoint()
           
            # except:
            #     print(set_path)
            #     print(audio_spk.shape, audio_lsn1.shape, audio_lsn2.shape, audio_lsn3.shape, audio_lsn4.shape)

            # pad audio
            audio_spk, audio_lsn1, audio_lsn2, audio_lsn3, audio_lsn4 = \
                self.pad_audios([audio_spk, audio_lsn1, audio_lsn2, audio_lsn3, audio_lsn4])

            melspec_lsn1, melspec_lsn2, melspec_lsn3, melspec_lsn4, melspec_spk = \
                self.get_melspecs([audio_lsn1, audio_lsn2, audio_lsn3, audio_lsn4, audio_spk])
            
            active_passive_bit = [self.check_audio(audio_lsn1), 
                                self.check_audio(audio_lsn2), 
                                self.check_audio(audio_lsn3), 
                                self.check_audio(audio_lsn4)]

            # load text
            with open(os.path.join(set_path, 'text_spk.txt'), 'r') as f:
                text_spk = f.read()
            with open(os.path.join(set_path, 'text_lsn1.txt'), 'r') as f:
                text_lsn1 = f.read()
            with open(os.path.join(set_path, 'text_lsn2.txt'), 'r') as f:
                text_lsn2 = f.read()
            with open(os.path.join(set_path, 'text_lsn3.txt'), 'r') as f:
                text_lsn3 = f.read()
            with open(os.path.join(set_path, 'text_lsn4.txt'), 'r') as f:
                text_lsn4 = f.read()

            uncond_sem = -1. * np.ones(self.max_motion_length)
            
            # uncond_mel = -90 * np.ones_like(melspec_spk)
            # uncond_mel[..., 40:45] = 0
            # data_dict['dndspk+' + set_name] = {
            #         'motion_spk': np.zeros_like(motion_spk),   
            #         'motions_lsn': [motion_spk],
            #         'melspec_spk': uncond_mel, 
            #         'melspecs_lsn': [melspec_spk],
            #         'text_spk': "-"*10,
            #         'texts_lsn': [text_spk],
            #         'audio_spk': np.zeros_like(audio_spk),
            #         'audios_lsn': [audio_spk],
            #         'active_passive_bit': [self.check_audio(audio_spk)],
            #         'sem_lsn': uncond_sem,
            #     }
            # # breakpoint()
            # name_list.append('dndspk+' + set_name)
            # dnd_names.append('dndspk+' + set_name)
            # length_list.append(motion_spk.shape[0])

            # if "test" in self.dnd_split_file:
            #     continue
            
            # data_dict['dnd+' + set_name] = {
            #     'motion_spk': motion_spk,   
            #     'motions_lsn': [motion_lsn1, motion_lsn2, motion_lsn3, motion_lsn4],
            #     'melspec_spk': melspec_spk,
            #     'melspecs_lsn': [melspec_lsn1, melspec_lsn2, melspec_lsn3, melspec_lsn4],
            #     'text_spk': text_spk,
            #     'texts_lsn': [text_lsn1, text_lsn2, text_lsn3, text_lsn4],
            #     'audio_spk': audio_spk,
            #     'audios_lsn': [audio_lsn1, audio_lsn2, audio_lsn3, audio_lsn4],
            #     'active_passive_bit': active_passive_bit,
            #     'sem_lsn': uncond_sem,
            # }
            # name_list.append('dnd+' + set_name)
            # dnd_names.append('dnd+' + set_name)
            # length_list.append(motion_spk.shape[0])
            if True: #self.check_audio(audio_lsn1).sum() != 0:
                data_dict['dnd+' + set_name + '_l1'] = {
                    'motion_spk': motion_spk,   
                    'motions_lsn': [motion_lsn1],
                    'melspec_spk': melspec_spk,
                    'melspecs_lsn': [melspec_lsn1],
                    'text_spk': text_spk,
                    'texts_lsn': [text_lsn1],
                    'audio_spk': audio_spk,
                    'audios_lsn': [audio_lsn1],
                    'active_passive_bit': [active_passive_bit[0]],'sem_lsn': uncond_sem,
                }
                name_list.append('dnd+' + set_name + '_l1')
                dnd_names.append('dnd+' + set_name + '_l1')
                length_list.append(motion_spk.shape[0])

            if self.check_audio(audio_lsn2).sum() != 0:
                data_dict['dnd+' + set_name  + '_l2'] = {
                    'motion_spk': motion_spk,   
                    'motions_lsn': [motion_lsn2],
                    'melspec_spk': melspec_spk,
                    'melspecs_lsn': [ melspec_lsn2],
                    'text_spk': text_spk,
                    'texts_lsn': [text_lsn2],
                    'audio_spk': audio_spk,
                    'audios_lsn': [ audio_lsn2],
                    'active_passive_bit': [active_passive_bit[1]],'sem_lsn': uncond_sem,
                }
                name_list.append('dnd+' + set_name + '_l2')
                dnd_names.append('dnd+' + set_name + '_l2')
                length_list.append(motion_spk.shape[0])

            if True: #self.check_audio(audio_lsn3).sum() != 0:
                data_dict['dnd+' + set_name  + '_l3'] = {
                    'motion_spk': motion_spk,   
                    'motions_lsn': [ motion_lsn3],
                    'melspec_spk': melspec_spk,
                    'melspecs_lsn': [ melspec_lsn3],
                    'text_spk': text_spk,
                    'texts_lsn': [ text_lsn3],
                    'audio_spk': audio_spk,
                    'audios_lsn': [ audio_lsn3],
                    'active_passive_bit': [active_passive_bit[2]],'sem_lsn': uncond_sem,
                }
                name_list.append('dnd+' + set_name + '_l3')
                dnd_names.append('dnd+' + set_name + '_l3')
                length_list.append(motion_spk.shape[0])

            if self.check_audio(audio_lsn4).sum() != 0:
                data_dict['dnd+' + set_name  + '_l4'] = {
                    'motion_spk': motion_spk,   
                    'motions_lsn': [ motion_lsn4],
                    'melspec_spk': melspec_spk,
                    'melspecs_lsn': [ melspec_lsn4],
                    'text_spk': text_spk,
                    'texts_lsn': [ text_lsn4],
                    'audio_spk': audio_spk,
                    'audios_lsn': [ audio_lsn4],
                    'active_passive_bit': [active_passive_bit[3]],'sem_lsn': uncond_sem,
                }
                name_list.append('dnd+' + set_name + '_l4')
                dnd_names.append('dnd+' + set_name + '_l4')
                length_list.append(motion_spk.shape[0])

            if "val" in self.dnd_split_file or "test" in self.dnd_split_file:
                continue
        
        # breakpoint()
        print("Length of loaded data from DnD: %d" % len(dnd_names))
        self.data_dict = data_dict
        self.nfeats = motion_lsn.shape[-1] # 63*3
        self.name_list = name_list
        print("Length of total loaded data: %d" % len(self.name_list))
            

       
    
    def beat_extract_text(self, text_path, frame_idx, length):
        # extract text data from text_path for the frame_idx to frame_idx + length

        text_data = parse_textgrid(text_path)
        seg_text = [[[float(s), float(e)],t] for s, e, t in zip(text_data['start'].tolist(), text_data['end'].tolist(), text_data['text'].tolist())]
        # breakpoint()
        start_sec = frame_idx / self.FPS
        end_sec = (frame_idx + length) / self.FPS
        text_indices = np.where((text_data['start'] >= start_sec) & (text_data['end'] <= end_sec))[0] # reduced window by one sec

        seg_text = [seg for seg in seg_text if seg[0][0] >= start_sec and seg[0][1] <= end_sec]
        seg_text = [[[seg[0][0] - start_sec, seg[0][1] - start_sec], seg[1]] for seg in seg_text]

        text_data = text_data['text'][text_indices]
        text_data = " ".join(text_data)

       

        return text_data, seg_text
    

    def beat_extract_sem(self, sem_path, frame_idx, length):
        sem_each_file = []

        sem_all_info = []
        try:
            sem_all = pd.read_csv(sem_path, 
                sep='\t', 
                names=["name", "start_time", "end_time", "duration", "score", "keywords"])
        except:
            return np.array([0.]*length)
        
        # we adopt motion-level semantic score here. 
        for i in range(frame_idx, frame_idx+length):
            found_flag = False
            for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                current_time = i/self.FPS + 0 # 0 is the time offset for all files we have in dataset
                if start<=current_time and current_time<=end: 
                    sem_each_file.append(score)
                    found_flag=True
                    break
                else: continue 
            if not found_flag: sem_each_file.append(0.)

        for v, (name, start, end, word) in enumerate(zip(sem_all['name'], sem_all['start_time'], sem_all['end_time'], sem_all['keywords'])):
            for k in range(frame_idx, frame_idx+length):
                current_time = k/self.FPS + 0
                if start<=current_time and current_time<=end: 
                    if "beat" in name:
                        class_name = "beat"
                        # break
                    elif "deictic" in name or "iconic" in name or "metaphoric" in name:
                        class_name = "semantic"
                    else:
                        break
                    # breakpoint()
                    if end > (frame_idx+length)/self.FPS:
                        chunk_end = length/self.FPS
                    else:
                        chunk_end = end - frame_idx/self.FPS

                    if start < frame_idx/self.FPS:
                        chunk_start = 0
                    else:
                        chunk_start = start - frame_idx/self.FPS
                    sem_all_info.append({
                        'name': class_name,
                        'start': chunk_start,
                        'end': chunk_end,
                        'word': word,
                    })
                    break
        
        return np.array(sem_each_file), sem_all_info


    def beat_extract_audio(self, filename, frame_idx, duration):
        start_sec = frame_idx*(1/self.FPS)
        duration_sec = duration*(1/self.FPS)

        audio, _ = librosa.load(filename, sr=self.SR)
        # print(filename, len(audio))
        audio_window_size = int(duration_sec*self.SR)
        # print(audio.shape[0] % audio_window_size)
        # audio = audio[:audio.shape[0] - audio.shape[0] % audio_window_size]
        # print(len(audio))
        audio_chunk = audio[int(start_sec*self.SR):int(start_sec*self.SR) + audio_window_size]
        
        assert len(audio_chunk) == int(duration_sec*self.SR), 'audio chunk length: {}, duration_sec: {} filename: {}, start {}, start sr {}, window {} full len {}'.format(len(audio_chunk), duration, filename, frame_idx, int(start_sec*self.SR), audio_window_size, len(audio))

        audio_chunk = librosa.util.normalize(audio_chunk)
        return audio_chunk
    
    # define a function to check if log db level is above threshold in a given audio and return a bool
    def check_audio(self, audio, threshold=-45):
        # if len(audio) == 0:
        #     return 0
        n_chunks = self.max_motion_length // 16
        audio_chunklen = int((16 / self.FPS)*self.SR)
        apb_bits = []
        for i in range(n_chunks):
            a_chunk = audio[i*audio_chunklen : (i+1)*audio_chunklen]
            audio_db = librosa.amplitude_to_db(a_chunk, ref=1)
        
            if np.max(audio_db) > threshold:
                apb_bits.append(1)
            else:
                apb_bits.append(0)

        return np.array(apb_bits)

    def pad_audios(self, audios):
        max_len = max([len(audio) for audio in audios])
        padded_audios = []
        for audio in audios:
            if len(audio) < max_len:
                pad = np.zeros(max_len - len(audio))
                print(len(pad))
                audio = np.concatenate((audio, pad))
            padded_audios.append(audio)
        return padded_audios
    

    def get_melspecs(self, audios):
        melspecs = []
        # breakpoint()
        # apb_list = active_passive_bit
        for audio in audios:
            # if len(audio) == 0:
            #     audio = np.zeros_like(audios[0])

            # if (audio == np.zeros_like(audio)).all():
            #     audio = np.ones_like(audio)
            # if apb == 0:
            #     audio = np.zeros_like(audio)
            
            melspec = librosa.feature.melspectrogram(
                y=audio, sr=self.SR, hop_length=self.HOP_LEN, n_mels=self.N_MELS)
            # breakpoint()
            melspec = librosa.power_to_db(melspec, ref=np.max)
            melspec = melspec.astype(np.float32)
            melspec = melspec.T
            
            melspecs.append(melspec)
        return melspecs

    
    def process_motion(self, motions):
        ret = []
        for motion in motions:
            # motion - (seq_len, joints, 3)
            # breakpoint()
            motion = motion[:, list(range(0,23)) +  list(range(24,44)) + list(range(46,66)), :] # njoints are 63 - 23 body 20 left and 20 right hands
            motion = motion/1000 # mm to m
            # reorder joints to make root joint first - already ordered in utterance making script for dnd dataset
            # motion = motion[:, [3] + list(range(0,3)) + list(range(4, motion.shape[1])) , :] 

            #  Put on floor
            floor_height = motion.min(axis=0).min(axis=0)[1]
            motion[:, :, 1] -= floor_height

            #  '''XZ at origin'''
            root_pos_init = motion[0]
            root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
            motion = motion - root_pose_init_xz

            # '''All initially face Z+'''
            r_hip, l_hip, sdr_r, sdr_l = self.face_joint_idx
            across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
            across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
            across = across1 + across2
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

            # forward (3,), rotate around y-axis
            forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
            # forward (3,)
            forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

            #     print(forward_init)

            target = np.array([[0, 0, 1]])
            root_quat_init = qbetween_np(forward_init, target)
            root_quat_init = np.ones(motion.shape[:-1] + (4,)) * root_quat_init

            motion_b = motion.copy()

            motion = qrot_np(root_quat_init, motion)

            # all joints root relative
            motion[:, 1:, :] = motion[:, 1:, :] - motion[:, :1, :] 

            # hands relative to wrist
            motion[:, 23:43, :] = motion[:, 23:43, :] - motion[:, [7], :] # 7 is wrist instead of 8
            motion[:, 43:, :] = motion[:, 43:, :] - motion[:, [11], :] # 11 is wrist instead of 12
            # motion[:, 23:, :] = motion[:, 23:, :] * 10
            motion = motion * 3 # all equal scale

            motion = motion.reshape(-1, self.njoints * 3)
            ret.append(motion)

        return ret
    
    def __len__(self):
        return len(self.name_list) # - self.pointer

    def __getitem__(self, idx):
        # breakpoint()
        
        data = self.data_dict[self.name_list[idx]]
        name = self.name_list[idx]

        dataset_name, path_name = name.split('+')
        # print("dn, pn ", dataset_name, path_name)

        motion_spk = data['motion_spk']
        motions_lsn = data['motions_lsn']
        melspec_spk = data['melspec_spk']
        melspecs_lsn = data['melspecs_lsn']
        audio_spk = data['audio_spk']
        audios_lsn = data['audios_lsn']
        text_spk = data['text_spk']
        texts_lsn = data['texts_lsn']
        active_passive_bit = data['active_passive_bit']
        sem_lsn = data['sem_lsn']
        sem_info = data['sem_info']

        if dataset_name == 'dnd':
            # active_idx = [idx for idx, x in enumerate(active_passive_bit) if x.sum() != 0]
            # if len(active_idx) == 0:
            #     active_idx = range(0, len(motions_lsn))
            name_idx = int(path_name[-1]) - 1
            set_path_name = path_name[:-3]
            
            # lsn_idx = random.sample(active_idx, 1)[0]
            lsn_idx = random.sample(range(0, len(motions_lsn)), 1)[0]
        else:
            lsn_idx = random.sample(range(0, len(motions_lsn)), 1)[0]
        
        # lsn_idx = 3

        motion_lsn = motions_lsn[lsn_idx]
        audio_lsn = audios_lsn[lsn_idx]
        melspec_lsn = melspecs_lsn[lsn_idx]
        text_lsn = texts_lsn[lsn_idx]
        active_passive_lsn = active_passive_bit[lsn_idx]

        if dataset_name == 'beat':
            spk_name = 'BEAT'
            lsn_name = path_name.split('/')[0].split('_')[1]
        elif dataset_name == 'dndspk':
            spk_name = 'dndspk'
            lsn_name = path_name.split('/')[1].split('_')[0].strip()
            # print(lsn_name)
        elif dataset_name == 'dnd':
            # check if any of speaker names are contained in name
            spk_name = [x for x in self.dnd_speaker_names if x in path_name]
            assert len(spk_name) == 1, 'speaker name not found in name: {}'.format(path_name)
            lsns = [x for x in self.dnd_speaker_names if x not in path_name]
            assert len(lsns) == 4, 'lsn names found in name: {}'.format(path_name)
            spk_name = spk_name[0]
            
            lsn_name = lsns[name_idx]
        else:
            raise ValueError('dataset name not found')

        # 
        lsn_id = self.speaker_names.index(lsn_name) + 1
        # print("spk, lsn, lsnid", spk_name, lsn_name, lsn_id)
        # print("spk_name", spk_name, "lsn_name", lsn_name, lsn_id, lsns, name)

        # for apb in enumerate(active_passive_lsn):
        #     if apb == 0:

        if dataset_name in ['dnd', 'dndspk'] :
            set_path = os.path.join(self.dnd_dataset_path, set_path_name)
            seg_lsn = None
            if os.path.exists(os.path.join(set_path, f'seg_lsn{name_idx+1}.txt')):
                with open(os.path.join(set_path, f'seg_lsn{name_idx+1}.txt'), 'r') as f:
                    seg_lsn = f.readlines()
                
                seg_lsn = [i.split('\t') for i in seg_lsn]
                
                seg_lsn = [[[float(i[0].strip()), float(i[1].strip())], i[2].strip()] for i in seg_lsn if i[2].strip() != '-']
                # print(seg_lsn)

            # print(text_lsn, "++", seg_lsn)
            # print('\n\n\n\n')

            seg_spk = None
            if os.path.exists(os.path.join(set_path, f'seg_spk.txt')):
                with open(os.path.join(set_path, f'seg_spk.txt'), 'r') as f:
                    seg_spk = f.readlines()
                
                seg_spk = [i.split('\t') for i in seg_spk]
                
                seg_spk = [[[float(i[0].strip()), float(i[1].strip())], i[2].strip()] for i in seg_spk if i[2].strip() != '-']

                
                # print(seg_spk)

            # print( seg_lsn, "++", seg_spk)
            # print('\n\n\n\n')
        else:
            seg_lsn = data['seg_lsn']
            seg_spk = data['seg_spk']

                

        if active_passive_lsn.sum() == 0:
            audio_lsn = np.zeros_like(audio_lsn)
            melspec_lsn = -80 + 0.01 * np.random.rand(*melspec_lsn.shape)
            text_lsn = '' #'-'*10
            # sem_lsn = np.zeros_like(sem_lsn)

        # if text_lsn == '':
        #     text_lsn = '-'*10 # represents no textual description (needed for discrimination against unconditional case)
        # if text_spk == '':
        #     text_spk = '-'*10
        text_lsn = text_lsn.strip()
        text_spk = text_spk.strip()

        
        
        
        m_length = motion_lsn.shape[0]

        # print("motion shape", motion_spk.shape, m_length)
        
        assert motion_spk.shape[0] == m_length, 'motion shape: {}, m_length: {}'.format(motion_spk.shape, m_length)

        
        if np.any(np.isnan(motion_spk)) or np.any(np.isnan(motion_lsn)):
            raise ValueError("nan in motion")

        # print("motion shape", motion_spk.shape, m_length, motion_lsn.shape, audio_spk.shape, audio_lsn.shape, melspec_spk.shape, melspec_lsn.shape, text_spk, text_lsn, active_passive_lsn, name, spk_name, lsn_name, lsn_id)
        if dataset_name in ['beat', 'dndspk']:
            other_mlsns = None
        elif dataset_name == 'dnd':
            other_mlsns = dict(zip(lsns[:lsn_idx] + lsns[lsn_idx+1 :], motions_lsn[:lsn_idx] + motions_lsn[lsn_idx+1 :]))
        else:
            raise ValueError('dataset name not found')

        # print(audio_spk.shape, [i.shape for i in audios_lsn]) 
        combined_audio = sum(audios_lsn) + audio_spk 
        # print(seg_lsn, seg_spk)
        return (
            motion_spk,
            m_length,
            motion_lsn,
            melspec_spk,
            melspec_lsn,
            audio_spk,
            audio_lsn,
            text_spk,
            text_lsn,
            active_passive_lsn,
            dataset_name + "/" + path_name,
            spk_name,
            lsn_name,
            lsn_id,
            other_mlsns,
            combined_audio,
            seg_lsn,
            seg_spk,
            sem_lsn,
            sem_info
        )




class MotionDataset(data.Dataset):
    def __init__(
        self,
        split_file,
        max_motion_length,
        min_motion_length,
        motion_rep,
        unit_length,
        dataset_path,
        debug=False,
        tiny=False,
        **kwargs,
    ):
        # # self.max_length = 512 # changed from 20 to 256
        # self.stage = kwargs['stage']
        self.motion_rep = motion_rep
        self.njoints = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length # this is used to convert motion length into multiples of unit_length
        self.njoints = 63
        self.face_joint_idx = kwargs['face_joint_idx']
        self.dataset_select = kwargs['dataset_select']
        # breakpoint()

        data_dict = {}
        beat_name_list = []
        dnd_name_list = []
        length_list = []

        beat_split_file = split_file[0]
        dnd_split_file = split_file[1]

        beat_dataset_path = dataset_path[0]
        dnd_dataset_path = dataset_path[1]

        # breakpoint()

        # load motion from BEAT dataset
        self.beat_split = np.loadtxt(beat_split_file, dtype=str)
        print(beat_split_file)
        assert self.motion_rep == 'pos', 'motion_rep should be pos for BEAT dataset'
        self.beat_motion_paths = [path for path in glob.glob(os.path.join(beat_dataset_path, '*/*.npy')) if 'euler' not in path]
        
        if debug:
            self.beat_split = self.beat_split[:10]
        if tiny:
            self.beat_split = self.beat_split[:5]
        
        self.beat_motion_paths.sort()
        self.beat_split.sort()
        if self.dataset_select == 'dnd': 
            self.beat_motion_paths = []

        # id_list = []
        
        print("Loading BEAT dataset from {}".format(beat_dataset_path))
        for motion_path in tqdm(self.beat_motion_paths):
            motion_name = os.path.basename(motion_path).replace('.npy', '')
            if motion_name not in self.beat_split:
                continue
            
            orig_motion = np.load(motion_path)

            # rescale frame rate to 25 frames per second to match with Dnd dataset
            # TODO: have a base frame rate for each dataset and map to 25 fps
            xp = np.arange(0, len(orig_motion), 120/25)

            if xp[-1] > len(orig_motion)-1:
                xp = xp[:-1]
            
            f = interp1d(np.arange(len(orig_motion)), orig_motion, axis=0)
            motion = f(xp) # (motion length, 228)
            # if motion_path == '/CT/mmughal/work/GestureSynth/BEAT/datasets/beat_english_v0.2.1/5/5_stewart_0_3_3_euler.npy':
            #     print(len(motion))
            #     breakpoint()

            if motion.shape[0] < self.max_motion_length:
                raise
                # continue
            # breakpoint()
            assert self.motion_rep == 'pos', 'motion_rep should be pos for BEAT dataset'
            # motion = motion[:, list(range(0,23)) +  list(range(24,44)) + list(range(46,66)), :] # njoints are 63 - 23 hands 20 left and 20 right hands
            # motion = motion/100 # cm to m
            motion = motion[:, [3] + list(range(0,3)) + list(range(4, motion.shape[1])) , :] 
            motion = motion * 10 # cm to mm to match with dnd dataset
            # reorder joints to make root joint first
            # motion = motion[:, [3] + list(range(0,3)) + list(range(4, motion.shape[1])) , :] 

            motion = motion[:motion.shape[0] - motion.shape[0] % self.max_motion_length] # (n_chunks * window_size, dim)
            motion_chunks = np.array_split(motion, motion.shape[0] // self.max_motion_length, axis=0) # (n_chunks, window_size, dim)
            
            # breakpoint()
            for idx, chunk in enumerate(motion_chunks):
                start_idx = idx * self.max_motion_length
                data_dict['beat/' + motion_name + '/' + str(idx)] = {
                            "motion": chunk,
                            "length": self.max_motion_length,
                            # "text_path": motion_path.replace('.npy', '.TextGrid').replace('_euler', ''),
                            # "audio_path": motion_path.replace('.npy', '.wav').replace('_euler', ''),
                            "start_idx": start_idx
                        }
                beat_name_list.append('beat/' + motion_name + '/' + str(idx))
                # length_list.append(len(chunk))

        print("Length of loaded data from BEAT: %d" % len(beat_name_list))

        # breakpoint()

        # load motion from DND dataset
        print('Loading DnD dataset from {}'.format(dnd_dataset_path))

        self.dnd_dataset_path = dnd_dataset_path
        print(dnd_split_file)
        self.dnd_split = np.loadtxt(dnd_split_file, dtype=str)
        assert self.motion_rep == 'pos', 'motion_rep should be pos for BEAT dataset'
        self.reaction_set_paths = glob.glob(os.path.join(dnd_dataset_path, '*/*'))
        
        # self.dnd_split = self.dnd_split[:500]
        if debug:
            self.dnd_split = self.dnd_split[:10]
        if tiny:
            self.dnd_split = self.dnd_split[:5] 
        # self.dnd_split = self.dnd_split[:50] 
        self.reaction_set_paths.sort()
        self.dnd_split.sort()
        if self.dataset_select == 'beat':
            self.reaction_set_paths = []
        # breakpoint()
        # self.reaction_set_paths = [os.path.join(dataset_path, "joints_03_02_23/chris_2195944_2201064")]*32

        # print("Loading DnD dataset from {}".format(self.dnd_dataset_path))
        for set_path in tqdm(self.reaction_set_paths):
            set_name = "/".join(set_path.split('/')[-2:])

            if set_name not in self.dnd_split:
                continue

            try:
                motion_spk = np.load(os.path.join(set_path, 'motion_spk.npy'))
                if motion_spk.shape[0] != self.max_motion_length:
                    print(motion_spk.shape[0], set_name)
                    continue

                motion_lsn1 = np.load(os.path.join(set_path, 'motion_lsn1.npy'))
                motion_lsn2 = np.load(os.path.join(set_path, 'motion_lsn2.npy'))
                motion_lsn3 = np.load(os.path.join(set_path, 'motion_lsn3.npy'))
                motion_lsn4 = np.load(os.path.join(set_path, 'motion_lsn4.npy'))
            except FileNotFoundError:
                continue

            motion_chunks = [motion_spk, motion_lsn1, motion_lsn2, motion_lsn3, motion_lsn4]

            for idx, chunk in enumerate(motion_chunks):
                data_dict['dnd/' + set_name + '/' + str(idx)] = {
                            "motion": chunk,
                            "length": self.max_motion_length,
                            # "text_path": motion_path.replace('.npy', '.TextGrid').replace('_euler', ''),
                            # "audio_path": motion_path.replace('.npy', '.wav').replace('_euler', ''),
                            "start_idx": 0
                        }
                dnd_name_list.append('dnd/' + set_name + '/' + str(idx))
                # length_list.append(len(chunk))
        print("Length of loaded data from DnD: %d" % len(dnd_name_list))

        # breakpoint()
        # start_motion_frame = {
        #     "joints_24_10_22": 185,
        #     "joints_28_10_22": 166,
        #     "joints_03_02_23": 88,
        #     "joints_24_02_23": 130,
        # }

        # self.person_paths = glob.glob(os.path.join(dnd_dataset_path, '*/*.csv')) #[:2]
        # self.person_paths.sort()

        # if debug or tiny:
        #     self.person_paths = self.person_paths[:1]


        # for person_path in tqdm(self.person_paths):
        #     df = pd.read_csv(person_path, header=None)
        #     frame_array = df.values[:, 1:].reshape(-1, 67, 3)

        #     session_name = os.path.dirname(person_path).split('/')[-1]
        #     motion_name = session_name + os.path.basename(person_path).replace('.csv', '')
        #     # print(session_name, person_path)
        #     start_idx = start_motion_frame[session_name]
        #     frame_array = frame_array[start_idx:]

        #     # Resample motion to 25 fps for joints_28_10_22
        #     if os.path.basename(session_name) == "joints_28_10_22":
        #         xp = np.arange(0, len(frame_array), 30/25)

        #         if xp[-1] > len(frame_array)-1:
        #             xp = xp[:-1]
                
        #         f = interp1d(np.arange(len(frame_array)), frame_array, axis=0)
        #         motion = f(xp)
        #     else:
        #         motion = frame_array

        #     motion = motion[:, list(range(0,23)) +  list(range(24,44)) + list(range(46,66)), :] # njoints are 63 - 23 hands 20 left and 20 right hands
        #     motion = motion/1000 # mm to m
        #     # reorder joints to make root joint first
        #     motion = motion[:, [3] + list(range(0,3)) + list(range(4, motion.shape[1])) , :] 

        #     motion = motion[:motion.shape[0] - motion.shape[0] % self.max_motion_length] # (n_chunks * window_size, dim)
        #     motion_chunks = np.array_split(motion, motion.shape[0] // self.max_motion_length, axis=0) # (n_chunks, window_size, dim)
            
        #     # Temporary splitting fix for DnD dataset
        #     if os.path.basename(beat_split_file) == 'train.txt':
        #         motion_chunks = motion_chunks[:int(len(motion_chunks)*0.8)]
        #     elif os.path.basename(beat_split_file) == 'val.txt':
        #         motion_chunks = motion_chunks[int(len(motion_chunks)*0.8):]

            
        #     # breakpoint()
        #     for idx, chunk in enumerate(motion_chunks):
        #         start_idx = idx * self.max_motion_length
        #         data_dict['dnd+' + motion_name + '+' + str(idx)] = {
        #                     "motion": chunk,
        #                     "length": self.max_motion_length,
        #                     # "text_path": motion_path.replace('.npy', '.TextGrid').replace('_euler', ''),
        #                     # "audio_path": motion_path.replace('.npy', '.wav').replace('_euler', ''),
        #                     "start_idx": start_idx
        #                 }
        #         name_list.append('dnd+' + motion_name + '+' + str(idx))

        # name_list, length_list = zip(
        #     *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        
        # self.id_list = id_list
        # self.length_arr = np.array(length_list)
        # breakpoint()
        for motion_id, motion_dict in data_dict.items():
            motion = motion_dict['motion']
            # breakpoint()
            #  Put on floor
            motion = motion[:, list(range(0,23)) +  list(range(24,44)) + list(range(46,66)), :] # njoints are 63 - 23 body 20 left and 20 right hands
            motion = motion/1000

            # reordering done in utterance script and loading script for dnd dataset

            floor_height = motion.min(axis=0).min(axis=0)[1]
            motion[:, :, 1] -= floor_height

            #  '''XZ at origin'''
            root_pos_init = motion[0]
            root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
            motion = motion - root_pose_init_xz

            # '''All initially face Z+'''
            r_hip, l_hip, sdr_r, sdr_l = self.face_joint_idx
            across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
            across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
            across = across1 + across2
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

            # forward (3,), rotate around y-axis
            forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
            # forward (3,)
            forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

            #     print(forward_init)

            target = np.array([[0, 0, 1]])
            root_quat_init = qbetween_np(forward_init, target)
            root_quat_init = np.ones(motion.shape[:-1] + (4,)) * root_quat_init

            motion_b = motion.copy()

            motion = qrot_np(root_quat_init, motion)

            # all joints root relative
            motion[:, 1:, :] = motion[:, 1:, :] - motion[:, :1, :] 

            # hands relative to wrist
            motion[:, 23:43, :] = motion[:, 23:43, :] - motion[:, [7], :] # 7 is wrist instead of 8
            motion[:, 43:, :] = motion[:, 43:, :] - motion[:, [11], :] # 11 is wrist instead of 12
            # motion[:, 23:, :] = motion[:, 23:, :] * 10
            motion = motion * 3 # all equal scale

            data_dict[motion_id]['motion'] = motion
            
        self.data_dict = data_dict
        self.nfeats = motion.shape[-2]*motion.shape[-1] # 63*3
        self.name_list = beat_name_list + dnd_name_list
        print("Length of loaded data: %d" % len(self.name_list))

        # self.mean = mean
        # self.std = std

        # breakpoint()
    
    def __len__(self):
        return len(self.name_list) # - self.pointer

    def __getitem__(self, idx):
        # breakpoint()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, chunk_start = data["motion"], data["length"], data["start_idx"]
        
        # uncomment below for random chunking
        # end_idx = random.randint(self.min_motion_length, m_length)
        # start_idx = random.randint(0, end_idx - self.min_motion_length)
        # chunk_start += start_idx
        # motion = motion[start_idx:end_idx]
        # m_length = end_idx - start_idx

        # print("motion length after first cut", len(motion), m_length, chunk_start, start_idx, end_idx)

        # m_length = (m_length // self.unit_length) * self.unit_length
        # # print("motion length after unit length stuff", len(motion), m_length, chunk_start, start_idx, end_idx)
        # s_idx = random.randint(0, len(motion) - m_length)
        # motion = motion[s_idx:s_idx + m_length]
        # chunk_start += s_idx
        # print("motion length after second cut", len(motion), m_length, chunk_start, start_idx, end_idx, s_idx)
        
        motion = motion.reshape(-1, self.njoints * 3)

        # print(motion.shape, m_length, self.name_list[idx])

        assert motion.shape[0] == m_length, 'motion shape: {}, m_length: {} at {}'.format(motion.shape, m_length, self.name_list[idx])

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")
        
        return (
            motion,
            m_length,
            self.name_list[idx],
        )
        # return caption, motion, m_length


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # dataset = TextAudio2MotionDataset('/CT/mmughal/work/GestureSynth/BEAT/datasets/beat_english_v0.2.1/train.txt',
    #                                  128,
    #                                  128, 
    #                                  'pos', 
    #                                  1, 
    #                                  '/CT/mmughal/work/GestureSynth/BEAT/datasets/beat_english_v0.2.1/', 
    #                                  debug=False,
    #                                  sample_rate=16000,
    #                                  hop_length=512,
    #                                  num_mels=80,
    #                                  fps=25
    #                                  )
    # # breakpoint()
    # # batch = dataset[0]
    # bs = 32
    # dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=11, drop_last=False, collate_fn=collate_fn)

    # for i, batch in enumerate(dataloader):
    #     print(i*bs, '/', len(dataset))
    #     print("motion shape: ", batch[0].shape)
    #     print("motion length: ", len(batch[1]))
    #     print("text data shape: ", len(batch[2]))
    #     print("audio data shape: ", batch[3].shape)
    #     print("melspec shape: ", batch[4].shape)
    #     print("name: ", batch[5])
    #     print("*"*100)

    # dataset = MotionDataset('/CT/mmughal/work/GestureSynth/BEAT/datasets/beat_english_v0.2.1/val.txt',
    #                                  128,
    #                                  128, 
    #                                  'pos', 
    #                                  1, 
    #                                  ['/CT/mmughal/work/GestureSynth/BEAT/datasets/beat_english_v0.2.1/', 
    #                                     '/CT/GroupGesture/work/DnD_first_three_recordings/joint_pos/joint_csv'],
    #                                  debug=False,
    #                                  face_joint_idx = [18, 13, 9, 5]
    #                                  )
    # # breakpoint()
    # # batch = dataset[0]
    # bs = 32
    # dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=11, drop_last=False, collate_fn=collate_fn)

    # for i, batch in enumerate(dataloader):
    #     print(i*bs, '/', len(dataset))
    #     print("motion shape: ", batch[0].shape)
    #     print("motion length: ", len(batch[1]))
    #     print("name: ", batch[2])
    #     print("*"*100)

    #     motion = batch[0].numpy()
    #     name = batch[2]
    #     for i, data in enumerate(zip(motion, name)):
    #         m, n = data
    #         p_keypoints3d = m.reshape(-1, dataset.njoints, 3)
    #         p_keypoints3d = p_keypoints3d / 3
    #         # p_keypoints3d[:, 23:, :] = p_keypoints3d[:, 23:, :] / 10
    #         p_keypoints3d[:, 43:, :] = p_keypoints3d[:, 43:, :] + p_keypoints3d[:, [12], :]
    #         p_keypoints3d[:, 23:43, :] = p_keypoints3d[:, 23:43, :] + p_keypoints3d[:, [8], :]
    #         p_keypoints3d[:, 1:, :] = p_keypoints3d[:, 1:, :] + p_keypoints3d[:, :1, :]

    #         np.save(f'/CT/GroupGesture/work/GestureSynth/reaction-latent-diffusion/results/test/{n}.npy', p_keypoints3d)

    #     break

    dataset = TextAudio2ReactionDataset(
        '/CT/GroupGesture/work/GestureSynth/test_ut_data/train.txt',
        128,
        128,
        'pos',
        1,
        '/CT/GroupGesture/work/GestureSynth/test_ut_data/',
        debug=True,
        face_joint_idx = [18, 13, 9, 5],
        sample_rate=16000,
        hop_length=512,
        num_mels=80,
        fps=25
    )
    # breakpoint()
    # batch = dataset[0]
    bs = 32
    # dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=11, drop_last=False, collate_fn=collate_fn)

    # sum_ = 0
    # sum2 = 0
    # sum3 = 0
    # sum4 = 0
    # sum5 = 0
    # for i, batch in enumerate(dataloader):
    #     # print(i*bs, '/', len(dataset))
        
    #     # # motion_spk,
    #     # #     motion_lsn,
    #     # #     melspec_spk,
    #     # #     melspecs_lsn,
    #     # #     audio_spk,
    #     # #     audio_lsn,
    #     # #     text_spk,
    #     # #     text_lsn,
    #     # #     active_passive_lsn,
    #     # #     length,
    #     # print("motion_spk shape: ", batch[0].shape)
    #     # print("motion_lsn shape: ", batch[2].shape)
    #     # print("melspec_spk shape: ", batch[3].shape)
    #     # print("melspecs_lsn shape: ", batch[4].shape)
    #     # print("audio_spk shape: ", batch[5].shape)
    #     # print("audio_lsn shape: ", batch[6].shape)
    #     # print("text_spk shape: ", batch[7])
    #     # print("text_lsn shape: ", batch[8])
    #     # print("active_passive_lsn shape: ", batch[9])
    #     # print("length: ", len(batch[1]))
    #     # print("name: ", batch[10])


    #     # print("*"*100)
    #     # break
    #     apb = batch["active_passive_lsn"]
    #     sum_ += sum(apb)
    #     for i in apb:
    #         if i == 1:
    #             if batch["text_lsn"][i] == '-':
    #                 sum2 += 1
    #             else:
    #                 sum3 += 1

    #         else:
    #             if batch["text_lsn"][i] == '-':
    #                 sum4 += 1
    #             else:
    #                 sum5 += 1


    # print("samples which have some audio in them \t", sum_, len(dataset), sum_/len(dataset)*100)
    # print("samples which have some audio in them but no text is transcribed \t", sum2, len(dataset), sum2/len(dataset)*100)
    # print("samples which have some audio in them but text is transcribed \t", sum3, len(dataset), sum3/len(dataset)*100)
    # print("samples which have silence in them and no text in them \t", sum4, len(dataset), sum4/len(dataset)*100)
    # print("samples which have silence in them and text is transcribed there (fault from whisper) \t", sum5, len(dataset), sum5/len(dataset)*100)



    for sample in dataset:
        motion_spk = sample[0]
        m_length = sample[1]
        motion_lsn  = sample[2]
        melspec_spk = sample[3]
        melspec_lsn = sample[4]
        audio_spk = sample[5]
        audio_lsn = sample[6]
        text_spk = sample[7]
        text_lsn = sample[8]
        active_passive_lsn = sample[9]
        name = sample[10]
        spk_name = sample[11]
        lsn_name = sample[12]
        lsn_id = sample[13]
