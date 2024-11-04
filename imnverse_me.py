import argparse
import torch
from torch._C import device
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import numpy as np
import controllable_generation_TV
from utils import cal_metric
from utils import restore_checkpoint, clear, batchfy, patient_wise_min_max, img_wise_min_max
from pathlib import Path
from models import utils as mutils
from models import ncsnpp 
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets
import time
# for radon
from physics.ct import CT, CT_LA
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
###############################################
# Configurations
###############################################
def angle_range(s):
    # Split the string on '-' and convert each part to an integer
    parts = list(map(int, s.split('-')))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("angle_range must be in the format 'start-end'")
    return tuple(parts)
parser = argparse.ArgumentParser(description='Controllable Generation CT with NCSN++')
parser.add_argument('--root', type=str, default='./data/CT/ind/256_sorted/L067',
                    help='Path to the directory containing the CT data')
parser.add_argument('--save-root', type=str, default='./results/AAPM_256_ncsnpp_continuous/sparseview_CT_ADMM_TV_total/m8/rho10/lambda0.04',
                    help='Path to the directory for saving generated samples')
parser.add_argument('--rho', type=int, default=10,
                    help='rho values')
parser.add_argument('--lamb', type=float, default=0.0001, 
                    help='lamb values')
parser.add_argument('--Nview', type=int, default=120,
                    help='Path to the directory for saving generated samples')
parser.add_argument('--snr', type=float, default=0.16,
                    help='Path to the directory for saving generated samples')                   
parser.add_argument('--size', type=int, default=256,
                    help='Path to the directory for saving generated samples')    
parser.add_argument('--ckpt_num', type=int, default=285, 
                    help='ckpt_num') 
parser.add_argument('--angle_range', type=angle_range, default=(0, 120),
                    help='Angle range for the radon transform, in the format "start-end"')
parser.add_argument('--iter_num', type=int, default=30,
                    help='iterition number') 
args = parser.parse_args()
snr = args.snr
lamb = args.lamb
iter_num= args.iter_num
rho = args.rho
root = Path(args.root)
save_root = Path(args.save_root)
Nview = args.Nview
size = args.size
problem = 'sparseview_CT_ADMM_TV_total'
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 1000
ckpt_num = args.ckpt_num
N = num_scales

vol_name = 'L067'

# Parameters for the inverse problem

det_spacing = 1.0

det_count = int((size * (2 * torch.ones(1)).sqrt()).ceil())

freq = 1

if sde.lower() == 'vesde':
    from configs.ve import AAPM_256_ncsnpp_continuous as configs
    ckpt_filename = f"exp/norm/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False

n_steps = 1

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)  ## model

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.parameters())

# Specify save directory for saving generated samples
save_root.mkdir(parents=True, exist_ok=True)

irl_types = ['input', 'recon', 'label', 'BP', 'sinogram']
for t in irl_types:
    if t == 'recon':
        save_root_f = save_root / t / 'progress'
        save_root_f.mkdir(exist_ok=True, parents=True)
    else:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)
def sort_key(x):
    try:
        return float(x.split(".")[0])
    except ValueError:
        return x
        # read all data
fname_list = os.listdir(root)
fname_list = sorted(fname_list, key=sort_key)
print(fname_list)
all_img = []

print("Loading all data")
for fname in tqdm(fname_list):
    just_name = fname.split('.')[0]
    img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True).squeeze())
    h, w = img.shape
    img = img.view(1, 1, h, w)
    all_img.append(img)
    plt.imsave(os.path.join(save_root, 'label', f'{just_name}.png'), clear(img), cmap='gray')
all_img = torch.cat(all_img, dim=0)
print(f"Data loaded shape : {all_img.shape}")

radon = CT_LA(img_width=h, radon_view=Nview, angle_range=args.angle_range , circle=False, device=config.device) #limit

predicted_sinogram = []
label_sinogram = []
img_cache = None

img = all_img.to(config.device)

pc_radon = controllable_generation_TV.get_pc_pdhg_TV_fan(sde,
                                                               predictor, corrector,
                                                               inverse_scaler,
                                                               snr=snr,
                                                               n_steps=n_steps,
                                                               probability_flow=probability_flow,
                                                               continuous=config.training.continuous,
                                                               denoise=True,
                                                               radon=radon,
                                                               save_progress=True,
                                                               save_root=save_root,
                                                               final_consistency=False,
                                                               img_shape=img.shape,
                                                               lamb_1=lamb,
                                                               rho=rho,
                                                               iter_num_1=iter_num,
                                                               view = Nview) # for pc radon
# Sparse by masking
sinogram = radon.A(img)
bp = radon.A_dagger(sinogram) #A_dagger AT
# Recon Image
start_time = time.time()
if args.method == 'admm':
    x = pc_radon(score_model, scaler(img), measurement=sinogram) # for pc radon
elif args.method == 'pdhg':
    x = pc_radon(score_model, scaler(img), measurement=img) # for pc radon
end_time = time.time()
img_cahce = x[-1].unsqueeze(0)
ssimes = []
psnres = []
count = 0
for i, recon_img in enumerate(x):
    plt.imsave(save_root / 'BP' / f'{count}.png', clear(bp[i]), cmap='gray')
    plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
    np.save(save_root / 'recon' / f'{count}.npy', clear(recon_img, normalize=False))
    plt.imsave(save_root / 'recon' / f'{count}.png', clear(recon_img), cmap='gray')
    recon = recon_img.detach().cpu().numpy().squeeze()
    gt = img[i].detach().cpu().numpy().squeeze()
    image1 = gt
    image2 = recon
    image1 = np.clip(image1, a_min=0, a_max=None)
    image2 = np.clip(image2, a_min=0, a_max=None)
    image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    ssim, ssim_map = ssim(image1, image2, full=True, data_range=1)
    psnr = peak_signal_noise_ratio(image1, image2, data_range=1)
execution_time = end_time - start_time
print(f"The execution time for pc_radon is: {execution_time} seconds; SSIM is: {ssim}; PSNR is: {psnr}")
