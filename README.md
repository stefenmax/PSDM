# Physics-informed Score-based Diffusion Model for Limited-angle Reconstruction of Cardiac Computed Tomography

Official PyTorch implementation of **PSDM**, "[Physics-informed Score-based Diffusion Model for Limited-angle Reconstruction of Cardiac Computed Tomography](https://arxiv.org/abs/2405.14770)". Code modified from [Diffusion-MBIR](https://github.com/HJ-harry/DiffusionMBIR).
This is just for the simulation purpose. The code for real experiment will update soon~

## Getting started


* Install dependencies
```
python = 3.7
torch = 1.13.1
scipy = 1.7.3
astra-toolbox
Operator Discretization Library (ODL)
sporco,tqdm, ninja,ml_collections
```

## PSDM reconstruction
Once you have the pre-trained weights and the test data set up properly, you may run the following scripts. Modify the parameters in the python scripts directly to change experimental settings.

```bash
conda activate your_conda
python imnverse_me.py
```

## Training
You may train the diffusion model with your own data by using e.g.
```bash
bash train_AAPM256.sh
```
You can modify the training config with the ```--config``` flag.

## Citation
If you find our work interesting, please consider citing

```
@article{han2024physics,
  title={Physics-informed Score-based Diffusion Model for Limited-angle Reconstruction of Cardiac Computed Tomography},
  author={Han, Shuo and Xu, Yongshun and Wang, Dayang and Morovati, Bahareh and Zhou, Li and Maltz, Jonathan S and Wang, Ge and Yu, Hengyong},
  journal={arXiv preprint arXiv:2405.14770},
  year={2024}
}
```
