import argparse

import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YS Project Training', add_help=add_help)
    parser.add_argument('--conffile', default='/home/yskim/PycharmProjects/lesion_drop_KI/config/lesion_drag_cls.py', type=str, help='')
    parser.add_argument('--batchsize', default=2, type=int, help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--gpuid', default='1', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--outputdir', default='./runs/train', type=str, help='path to save outputs')
    parser.add_argument('--foldername', default='debug', type=str, help='experiment name, saved to output_dir/name')
    parser.add_argument('--loss', default='focal', type=str, help='[focal | ce | mse | mae | rmse | dice]')
    parser.add_argument('--lossredu', default='sum', type=str, help='')
    parser.add_argument('--inchannel', default=3, type=int, help='')
    parser.add_argument('--datajson', default='splitted_data', type=str, help='')
    parser.add_argument('--mode', default='patch_cls', type=str, help='')
    parser.add_argument('--netmode', default='wimg', type=str, help='')
    parser.add_argument('--patchsize', default=512, type=int, help='')
    #
    parser.add_argument('--procname', default='HRLEE:drag_cls_debug', type=str, help='')
    return parser
