import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--spokes', type=int, metavar='', required=True)
parser.add_argument('-g', '--gpu', type=int, metavar='', required=True)
parser.add_argument('-t', '--tv_weight', type=float, metavar='', required=False, default=0.02)
parser.add_argument('-l', '--lr_weight', type=float, metavar='', required=False, default=0.0002)
parser.add_argument('-st', '--stv_weight', type=float, metavar='', required=False, default=0) # Just in case
parser.add_argument('-n', '--neuron', type=int, metavar='', required=False, default=128)
parser.add_argument('-ly', '--layers', type=int, metavar='', required=False, default=5)
parser.add_argument('-hs', '--log2_hashmap_size', type=int, metavar='', required=False, default=24)
parser.add_argument('-ls', '--per_level_scale', type=float, metavar='', required=False, default=2.0)
parser.add_argument('-e', '--epochs', type=int, metavar='', required=False, default=1600)
parser.add_argument('-m', '--mask', action='store_true', required=False)
parser.add_argument('-r', '--relL2', action='store_true', required=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch
import datetime
import h5py
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from utils import fftnc, ifftnc, coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from scipy import io
import sigpy.mri as mr
from model import INR

params = {
    'n_levels': 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": args.log2_hashmap_size,
    "base_resolution": 16,
    "per_level_scale": args.per_level_scale,
    'lr': 0.001,
    "n_neurons": args.neuron,
    "n_hidden_layers": args.layers,
    "tv_weight": args.tv_weight,
    "lr_weight": args.lr_weight,
    "stv_weight": args.stv_weight,
    "epochs": args.epochs, 
    "mask": args.mask,
    "relL2": args.relL2
}
print(params)

# Important Constants
GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))  # GoldenAngle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-3
summary_epoch = 50
spoke_num = args.spokes
epoch = params['epochs']
relL2_eps = 1e-4
scale = 2


log_path = './log_cmr/spoke{}_{}'.format(spoke_num, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S')))
path_checker(log_path)
writer = SummaryWriter(log_path)

# Import and Preprocess Data
mat_path = './test_cardiac.mat'
with h5py.File(mat_path, 'r') as f:
    img_full = f['img'][:]
    smap = f['smap'][:]
img_full = torch.as_tensor(img_full).to(device)
smap = torch.as_tensor(smap).to(device)
frames_full = img_full.shape[0]
coil_num = img_full.shape[1]
grid_size = img_full.shape[-1]
spoke_length = grid_size * 2
ktraj_full = gen_traj(GA, spoke_length, frames_full * spoke_num).reshape(2, frames_full, -1).transpose(1, 0)
ktraj = ktraj_full[::scale, ...]
frames = ktraj.shape[0]

img_gt_full = coil_combine(img_full, smap)
scale_factor = torch.abs(img_gt_full).max()
img_gt_full /= scale_factor
img_gt = img_gt_full[::scale, ...]
dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([spoke_num, 1]).to(device)
nufft_op = NUFFT(ktraj, dcomp, smap)
kdata = nufft_op.forward(img_gt).reshape([frames, coil_num, spoke_num, spoke_length])

# Build Model and Loss
inr = INR(nufft_op, params, lr, relL2_eps)
pos, pos_dense_t = inr.build_tsr_pos(grid_size, frames, scale)


psnr = 0.0
ssim = 0.0
time_usage = 0.0
epoch_loop = tqdm(range(epoch), total=epoch, leave=True)
for e in epoch_loop:

    # Training
    intensity, delta_time = inr.train(pos, kdata, e)
    time_usage += delta_time
    epoch_loop.set_description("[Train] [Lr:{:5e}]".format(inr.scheduler.get_last_lr()[0]))
    epoch_loop.set_postfix(dc_loss=inr.dc_loss.item(), tv_loss=inr.tv_loss.item(), max=torch.abs(intensity).max().item(),
                           lowrank_loss=inr.lowrank_loss.item())
    writer.add_scalar('loss_train', inr.loss_train, e + 1)

    # Infering
    if (e + 1) % summary_epoch == 0:
        with torch.no_grad():
            intensity, psnr_tmp, ssim_tmp = inr.infer(pos_dense_t, img_gt_full, smap, tscale=scale)
        io.savemat(log_path + '/proposed_{}.mat'.format(e+1),
                    {'img_proposed': intensity.cpu().numpy()})
        visual_mag(intensity,
            log_path + '/proposed_{}_{}_abs_{}.png'.format(spoke_num, frames, e+1))
        visual_err_mag(intensity, img_gt_full, log_path + '/proposed_{}_{}_abs_err_{}.png'.format(spoke_num, frames, e+1))
        writer.add_scalar('psnr', psnr_tmp, e + 1)
        writer.add_scalar('ssim', ssim_tmp, e + 1)
        if psnr_tmp > psnr:
            psnr = psnr_tmp

# Summary
print('Best PSNR: {:.4f}'.format(psnr))
print('Time Consumption: {:.2f}s'.format(time_usage))