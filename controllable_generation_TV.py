import functools
import time

import torch
from numpy.testing._private.utils import measure
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.fft import fft2, ifft2, fftshift, ifftshift
from models import utils as mutils
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
#from utils import fft2, ifft2, fft2_m, ifft2_m
from physics.ct import *
from utils import show_samples, show_samples_gray, clear, clear_color, clear_window, batchfy
import odl


class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass
class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb


def prox_l21(src, lamb, dim):
    """
    src.shape = [448(z), 1, 256(x), 256(y)]
    """
    weight_src = torch.linalg.norm(src, dim=dim, keepdim=True)
    weight_src_shrink = shrink(weight_src, lamb)

    weight = weight_src_shrink / weight_src
    return src * weight


def shrink(weight_src, lamb):
    return torch.sign(weight_src) * torch.max(torch.abs(weight_src) - lamb, torch.zeros_like(weight_src))


## Revised version
def frequency_fusion(image1, image2):
    f1 = fft2(image1)
    f2 = fft2(image2)
    
    # Shift the zero frequency component to the center
    f1_shift = fftshift(f1)
    f2_shift = fftshift(f2)
    
    # Initialize the fused frequency domain with the first image's frequencies
    fused_freq = np.copy(f1_shift)
    rows, cols = image1.shape
    mask = np.zeros_like(image1)
    mask[:rows//2, cols//2:] = 1  
    mask[rows//2:, :cols//2] = 1  
    fused_freq = f1_shift + f2_shift * mask
    fused_freq_shift = ifftshift(fused_freq)
    fused_image = ifft2(fused_freq_shift)
    
    return np.real(fused_image)



def get_pc_pdhg_TV_fan(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                             final_consistency=False, img_shape=None, lamb_1=5, rho=10,iter_num_1=30, view=90):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)
    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None,
                 norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean


    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn
        

    
    def CS_routine(x1, measurement, niter, lambd, view, siter):
        reco_space = odl.uniform_discr(
        min_pt=[-50, -50], max_pt=[50, 50], shape=[256, 256], dtype='float32')
        detector_partition = odl.uniform_partition(-70, 70, 363) # 256 256 363
        limit_view = view
        angle_partition = odl.uniform_partition(0+np.pi/2, limit_view * np.pi/180+np.pi/2, limit_view) 
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=1000, det_radius=100)
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
        fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann')
        discr_phantom = measurement.cpu().numpy().squeeze()
        data = ray_trafo(discr_phantom)
        fbpimage = fbp_op(data)
        fbpimage = np.array(fbpimage, dtype=np.float32)

        gradient = odl.Gradient(reco_space)
        # Column vector of two operators
        op = odl.BroadcastOperator(ray_trafo, gradient)
        f = odl.solvers.ZeroFunctional(op.domain)
        l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)  

        l1_norm = lambd * odl.solvers.L1Norm(gradient.range)  # 0.015 ori iso/

        # Combine functionals, order must correspond to the operator K
        g = odl.solvers.SeparableSum(l2_norm, l1_norm)

        # --- Select solver parameters and solve using PDHG --- #

        # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
        op_norm = 1.1 * odl.power_method_opnorm(op)
        tau = 1 / op_norm  # Step size for the primal variable 1.0
        sigma = 1 / op_norm  # Step size for the dual variable 1.0
        x1 = x1.cpu().numpy().squeeze()
        
        ## Insert fourier fusion prior
        if 400 < siter < 800:
            x1 = frequency_fusion(x1,fbpimage)

        
        x1 = op.domain.element(x1)
        x = x1
        # Run the algorithm
        odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)
        x = np.array(x, dtype=np.float32)
        x = x.copy()
        x = torch.from_numpy(x)
    
        x = x.unsqueeze(0).unsqueeze(0).to(measurement.device)
        x_mean = x
        return x, x_mean
    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, lambd, iter_num, view, measurement=None, siter=None):
            with torch.no_grad():
                x, x_mean = CS_routine(x, measurement, niter=iter_num, lambd = lambd, view=view, siter=siter)
                return x, x_mean
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None, lambd = lamb_1, iter_num = iter_num_1, view = view):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 12)
                # 2. Run PC step for each batch
                x_agg = list()

                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run PDHG TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run PDHG TV
                x, x_mean = mc_update_fn(x,lambd, iter_num, view, measurement=measurement,siter=i)

                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        np.save(save_root / 'recon' / 'progress'/ f'progress{i}.npy', clear_window(x_mean))
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear_window(x_mean[0:1]), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon
