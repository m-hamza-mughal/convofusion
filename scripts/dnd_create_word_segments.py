import glob
import librosa
import os
import whisper
import numpy as np
from tqdm import tqdm

# from multiprocessing import Pool

model = whisper.load_model("medium.en")

def transcribe(audio_path):
    audio_array, sr = librosa.load(audio_path, sr=16000)
    # breakpoint()
    audio_np = audio_array.astype(np.float32)
    # audio_np /= np.iinfo(audio_array.typecode).max 
    # breakpoint()
    result = model.transcribe(audio_np, word_timestamps=True)
    all_words = []
    for i, seg in enumerate(result["segments"]):
        words = seg["words"]
        for w in words:
            all_words.append([w['start'], w['end'], w['word']])

    del audio_array, audio_np
    return result["text"], all_words


def process_files(af):
    text, segments = transcribe(af)
    if text is None:
        text = ""
        segments = []
    # breakpoint()
    with open(os.path.join(os.path.dirname(af), 'seg_' + os.path.basename(af).split('_')[-1][:-4] + '.txt'), 'w') as f:
        seg_lines = [str(start) + '\t' + str(end) + '\t' + txt + '\n' for start, end, txt in segments]
        f.writelines(seg_lines)
    # print(af, text)

if __name__=='__main__':
    audio_files = glob.glob('./datasets/utterance_dataset_30sec/*/*/*.wav')
    
    print("Number of files: ", len(audio_files))
    for af in tqdm(audio_files):
        process_files(af)

    print("Done")

