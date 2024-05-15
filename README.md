# ConvoFusion


### [ConvoFusion: Multi-Modal Conversational Diffusion for Co-Speech Gesture Synthesis](https://vcai.mpi-inf.mpg.de/projects/ConvoFusion/)

### [Project Page](https://vcai.mpi-inf.mpg.de/projects/ConvoFusion/) | [Arxiv](https://arxiv.org/abs/2403.17936) - CVPR 2024


<p float="center">
  <img src="misc/TEASER.png" />
</p>

## 🚩 Updates

- [15.05.2024] initial code release

## Initial Steps

<!-- <details> -->
  <summary><b>Setup and download</b></summary>
  
### 1. Conda environment

```
conda env create --name convofusion --file=environment.yml 
```

#### OR

```
conda create python=3.9 --name convofusion
conda activate convofusion
```

Install the packages in `requirements.txt` and install [PyTorch 2.1.2](https://pytorch.org/)

```
pip install -r requirements.txt
```

Download 

<!-- ```
python -m spacy download en_core_web_sm
``` -->

### 2. Dependencies

Will add more instructions on how to download dependencies for training. 

### 3. Pre-train model

Download model folders from [this link](https://nextcloud.mpi-klsb.mpg.de/index.php/s/PWnL4HA3wQ7nJnZ), extract zip file and place both folders in `experiments/convofusion/`

<!-- </details> -->

<!-- ## ▶️ Demo -->

<!-- <details>
  <summary><b>Gesture Generation</b></summary>

add demo script
</details> -->

## Train your own models

<!-- <details> -->
  <summary><b>Training guidance</b></summary>

### 1. Prepare the datasets

Setup [BEAT](https://pantomatrix.github.io/BEAT/) and [DnD Group Gesture Dataset](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.IPFYCC).
Instructions on processing data (BVH to joint conversions).

### 2.1. Train VAE model

Please first check the parameters in `configs/config_vae_beatdnd.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command:

```
python -m train --cfg configs/config_vae_beatdnd.yaml --cfg_assets configs/assets.yaml --batch_size 128 --nodebug
```

### 2.2. Train latent diffusion model

Please update the parameters in `configs/config_cf_beatdnd.yaml`, e.g. `NAME`,`DEBUG`,`PRETRAINED_VAE` (change to your `latest ckpt model path` in previous step)

Then, run the following command:

```
python -m train --cfg configs/config_cf_beatdnd.yaml --cfg_assets configs/assets.yaml --batch_size 32 --nodebug
```

### 3. Get the model outputs on test set

Please first put the tained model checkpoint path to `TEST.CHECKPOINT` in `configs/config_cf_beatdnd.yaml` or the config of your experiment folder `/path/to/trained-model/folder/config.yaml`.

Then, run the following command:

```
python -m test --cfg /path/to/trained-model/folder/config.yaml --cfg_assets ./configs/assets.yaml
```

### 4. Visualization

Utilize and tweak `visualize.py` script in scripts folder to visualize joint prediction. The results folder will be created after you run `test.py`

```
python visualize.py --src_dir /path/to/results/folder/
```

### 5. Quantitative Evaluation

We provide scripts for quantitative evaluation in `quant_eval` folder for both monadic and dyadic tasks. These scripts require the generated results folder containing predicted and GT npy motion files.

<!-- </details> -->


## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@InProceedings{mughal2024convofusion,
title = {ConvoFusion: Multi-Modal Conversational Diffusion for Co-Speech Gesture Synthesis},
author = {Muhammad Hamza Mughal and Rishabh Dabral and Ikhsanul Habibie and Lucia Donatelli and Marc Habermann and Christian Theobalt},
booktitle={Computer Vision and Pattern Recognition (CVPR)},
year={2024}
}
```

## Acknowledgments

This repository is based on the awesome [MLD repository](https://github.com/ChenFengYe/motion-latent-diffusion). Please check out their repository for further acknowledgements of code which they use. We would also like acknowledge the authors of [BEAT](https://pantomatrix.github.io/BEAT/), [Attend-and-Excite](https://yuval-alaluf.github.io/Attend-and-Excite/), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [PhysCap](https://vcai.mpi-inf.mpg.de/projects/PhysCap/) & [MoFusion](https://vcai.mpi-inf.mpg.de/projects/MoFusion/) since our code is also based on them.
<br>
This work was supported by the ERC Consolidator Grant 4DReply (770784). We also thank Andrea Boscolo Camiletto & Heming Zhu for help with rendering and visualizations and Christopher Hyek for designing the game for the dataset

## License

This code is distributed under the terms of the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
This project is only for research or education purposes, and not freely available for commercial use or redistribution.


Note that our code depends on other libraries, including PyTorch3D, and uses dataset like BEAT which each have their own respective licenses that must also be followed.


