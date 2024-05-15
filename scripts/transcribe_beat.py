import whisper
import glob
import os
import json
from tqdm import tqdm

def transcribe(model, audio_path):
    result = model.transcribe(audio_path, word_timestamps=True)
    return result

def main(src_dir, 
         out_dir):
    model = whisper.load_model("medium.en", download_root='/CT/mmughal/work/GestureSynth/whisper_ASR/model_cache/')

    audio_files = glob.glob(os.path.join(src_dir, "*/*.wav"))
    for audio_path in tqdm(audio_files):
        # breakpoint()
        output = transcribe(model, audio_path)

        dest_jsonpath = os.path.join(out_dir, '/'.join(audio_path.split('/')[-2:]).replace(".wav", ".json"))
        os.makedirs(os.path.dirname(dest_jsonpath), exist_ok=True)

        with open(dest_jsonpath, "w") as f:
            json.dump(output, f)

        

if __name__ == "__main__":
    src_dir = "/CT/mmughal/work/GestureSynth/BEAT/datasets/beat_english_v0.2.1/"
    out_dir = "/CT/mmughal/work/GestureSynth/BEAT/datasets/whisper_transcription/"

    main(src_dir, out_dir)
    

