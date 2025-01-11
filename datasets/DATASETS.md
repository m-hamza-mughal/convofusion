> Datasets are put in this folder

Download `beat_english_v0.2.1` folder from BEAT's [google drive](https://drive.google.com/drive/folders/1EKuWH8q178QOtFUYaNohdkZbBHQYAmhL). 

Download processed DnD joints files from [this link](https://edmond.mpg.de/file.xhtml?fileId=264610&version=3.0). 


# Preprocessing BEAT

Install [PyMO](https://github.com/omimo/PyMO) and scikit-learn for the conversion script of BEAT skeleton to DnD skeleton.
Run following script by changing the paths according to your setup. 
```
python scripts/beat_getjoints.py
```
This script also contains joint name list which corresponds to 67 joints (same as the preprocessed DnD dataset - see below)


# Splitting DnD Dataset.
To run the code, you would need to download the processed version (joint positions) for DnD GroupGesture Dataset. 
Download  from the [link](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.IPFYCC) and unzip it. 

Then you would need to run following script which chunks out the DnD Gesture datasets in 128 frame windows(5.12sec) and transcribes the audio according to each chunk, giving us text annotations and segments
```
python scripts/dnd_make_utterance_dataset.py
```

# Information on Processed Version of Tracking data

Similar to BVH data, the folder contains four sessions and each session folder contains a wav file and csv file. Extract the folder in `./datasets/` directory.
The csv file contains joint keypoint positions (x,y,z) and The wav files contains clean audio of each speaker.

# Explanation on Number of joints

The conversion scripts above result in 67 joints (23 body joints and 44 hand joints).
However, joints like `LeftHand` and `LeftHandMid` are repeated in body and hands (refer to `scripts/beat_getjoints.py` for full joint list for our skeleton)
So we remove the repeated joints for hands, which results in 63 joints (mentioned in the paper).


