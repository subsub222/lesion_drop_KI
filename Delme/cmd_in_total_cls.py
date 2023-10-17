import argparse

import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YS Project Training', add_help=add_help)
    parser.add_argument('--conffile', default='./config/lesion_total_cls.py', type=str, help='')
    parser.add_argument('--batchsize', default=2, type=int, help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--gpuid', default='1', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--outputdir', default='./runs/train', type=str, help='path to save outputs')
    parser.add_argument('--foldername', default='tiscls_resnet', type=str, help='experiment name, saved to output_dir/name')
    parser.add_argument('--loss', default='focal', type=str, help='[focal | ce | mse | mae | rmse | dice]')
    parser.add_argument('--lossredu', default='sum', type=str, help='')
    parser.add_argument('--inchannel', default=3, type=int, help='')
    parser.add_argument('--inputsize', default=1024, type=int, help='')
    parser.add_argument('--datajson', default='splitted_data', type=str, help='')
    parser.add_argument('--mode', default='total_cls', type=str, help='')
    # parser.add_argument('--gain', action='store_true', help='')
    parser.add_argument('--gain', default=True, type=bool, help='')
    parser.add_argument('--gaintype', default='meta', type=str, help='meta')
    #
    # gain = True, gaintype = meta, mode = tiscls --> In datasets, data[task + _mask] will be used.
    # gain = True, gaintype = meta, mode = total_cls --> In datasets, data[task] will be used.
    # gain = True, gaintype = per, mode = tiscls --> In datasets, data[task + _mask] will be used.
    #
    return parser
