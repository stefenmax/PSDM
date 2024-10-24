# Physics-informed Score-based Diffusion Model for Limited-angle Reconstruction of Cardiac Computed Tomography

Official PyTorch implementation of **PSDM**, "[Physics-informed Score-based Diffusion Model for Limited-angle Reconstruction of Cardiac Computed Tomography](https://arxiv.org/abs/2405.14770)". Code modified from [Diffusion-MBIR](https://github.com/HJ-harry/DiffusionMBIR).


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
conda activate diffusion-mbir
python inverse_problem_solver_AAPM_3d_total.py
python inverse_problem_solver_BRATS_MRI_3d_total.py
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
@InProceedings{chung2023solving,
  title={Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models},
  author={Chung, Hyungjin and Ryu, Dohoon and McCann, Michael T and Klasky, Marc L and Ye, Jong Chul},
  journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
