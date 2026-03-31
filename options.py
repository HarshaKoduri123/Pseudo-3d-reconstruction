# options.py

import argparse
from datetime import datetime

def set_argparse_defs(parser):
    parser.set_defaults(accelerator='gpu')
    parser.set_defaults(devices=1)
    parser.set_defaults(num_sanity_val_steps=0)
    parser.set_defaults(progress_bar_refresh_rate=5)
    parser.set_defaults(log_every_n_steps=10)
    parser.set_defaults(deterministic=False)
    return parser

def add_argparse_args(parser):
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=10)
    parser.add_argument('--valid_batch_size', dest='valid_batch_size', type=int, default=3)
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--lr_start', dest='lr_start', type=float, default=0.0001)
    parser.add_argument('--lr_param', dest='lr_param', type=float, default=0.9)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--decay', dest='decay', type=float, default=0.0000)
    parser.add_argument('--nesterov', dest='nesterov', action='store_true')
    parser.add_argument('--optim', dest='optim', default='adam')
    parser.add_argument('--monitor', dest='monitor', default='val_metric0')
    parser.add_argument('--monitor_mode', dest='monitor_mode', default='max')
    parser.add_argument('--network', dest='network', default='unet2d_240')
    parser.add_argument('--networks', nargs='+', dest='networks', default=['unet2d_240'])
    parser.add_argument('--no_global_skip', dest='global_skip', action='store_false')
    parser.add_argument('--no_skip', dest='skip', action='store_false')
    parser.add_argument('--trainee', dest='trainee', default='segment')
    parser.add_argument('--loss', dest='loss', default='cce_loss')
    parser.add_argument('--train_metrics', nargs='+', dest='train_metrics', default=[])
    parser.add_argument('--input_metrics', nargs='+', dest='input_metrics', default=[])
    parser.add_argument('--valid_metrics', nargs='+', dest='valid_metrics', default=[])
    parser.add_argument('--tests_metrics', nargs='+', dest='tests_metrics', default=[])
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--padding', dest='padding', type=int, default=0)
    parser.add_argument('--padding_mode', dest='padding_mode', nargs='+', default=['circular', 'reflect', 'reflect'])
    parser.add_argument('--remarks', dest='remarks', default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--drop', dest='drop_rate', type=float, default=0.0)
    parser.add_argument('--schedule', dest='schedule', default='flat')
    parser.add_argument('--fraction', dest='fraction', default=1.0, type=float)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--weighted', dest='weighted', action='store_true')
    parser.add_argument('--save_train_output_every', dest='save_train_output_every', type=int, default=0)
    parser.add_argument('--save_valid_output_every', dest='save_valid_output_every', type=int, default=0)
    parser.add_argument('--save_tests_output_every', dest='save_tests_output_every', type=int, default=0)
    parser.add_argument('--save_weight_every', dest='save_weight_every', type=int, default=100)
    parser.add_argument('--validate', dest='validate', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--load', dest='load', default='')
    parser.add_argument(
        '--dataset', dest='dataset', default='sandstone',  
        help='Dataset to use: sandstone')
    parser.add_argument(
        '--dataroot', dest='dataroot',
        default=r'C:\Users\PRASANTH\3D-Reconstruction\dataset\DRP-317',
        help='Path to DRP-317 folder containing all sandstone samples')
    parser.add_argument(
        '--patch_size', dest='patch_size', type=int, default=64,
        help='Cubic patch side length for training (e.g. 64 = 64x64x64)')
    parser.add_argument(
        '--patch_stride', dest='patch_stride', type=int, default=32,
        help='Stride between patch origins (32 = 50%% overlap)')
    parser.add_argument(
        '--subsample_factors', dest='subsample_factors', nargs='+', type=int,
        default=[5, 10, 15],
        help='Sparsity factors to simulate: keep every Nth slice (e.g. 5 10 15)')
    parser.add_argument(
        '--input_type', dest='input_type', default='filtered',
        choices=['filtered', 'grayscale'],
        help='Which CT volume to use as input: filtered (denoised) or grayscale (raw)')
    parser.add_argument(
        '--max_patches', dest='max_patches', type=int, default=None,
        help='Max patches per volume — set e.g. 100 for quick dev run')

    return parser