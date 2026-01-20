import argparse
import os
import json
from datautils import load_UCR, load_HAR, load_mat, load_txt, load_txt_regress, load_txt_valid, load_txt_all
import torch
parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--save_path', type=str, default='exp/mono_full_EFSCV_serotonin/test')
parser.add_argument('--dataset', type=str, default='txt_all')
parser.add_argument('--UCR_folder', type=str, default='mono_full_EFSCV')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--train_batch_size', type=int, default=128)
# parser.add_argument('--valid_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)

# model args
parser.add_argument('--resume', type=int, default=0, help='Whether to resume training from a checkpoint')
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--eval_per_steps', type=int, default=16)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--alpha', type=float, default=5.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--vocab_size', type=int, default=192)
parser.add_argument('--wave_length', type=int, default=8)
parser.add_argument('--mask_ratio', type=float, default=0.6)
parser.add_argument('--reg_layers', type=int, default=4)
parser.add_argument('--num_targets', type=int, default=2)
parser.add_argument('--target', type=str, default='both')

# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=1)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epoch_pretrain', type=int, default=51) # Number of epochs you want + 1
parser.add_argument('--num_epoch', type=int, default=51) # Number of epochs you want + 1
parser.add_argument('--load_pretrained_model', type=int, default=0)

args = parser.parse_args()
args.resume = bool(args.resume)
if args.data_path is None:
    if args.dataset == 'ucr':
        Train_data_all, Train_data, Test_data = load_UCR(folder=args.UCR_folder)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'har':
        Train_data_all, Train_data, Test_data = load_HAR()
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'mat':
        Train_data_all, Train_data, Test_data = load_mat()
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'txt':
        Train_data_all, Train_data, Test_data, label_dict  = load_txt()
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'txt_regress':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_txt_regress(path, folder=args.UCR_folder)
    elif args.dataset == 'txt_valid':
        path = args.data_path
        Train_data, Valid_data, Test_data = load_txt_valid(path, folder=args.UCR_folder)
    elif args.dataset == 'txt_all':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_txt_all(path, folder=args.UCR_folder)
else:
    if args.dataset == 'ucr':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_UCR(path, folder=args.UCR_folder)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'har':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_HAR(path)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'mat':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_mat(path)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'txt':
        path = args.data_path
        Train_data_all, Train_data, Test_data, label_dict = load_txt(path, folder=args.UCR_folder)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'txt_regress':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_txt_regress(path, folder=args.UCR_folder)
    elif args.dataset == 'txt_valid':
        path = args.data_path
        Train_data, Valid_data, Test_data = load_txt_valid(path, folder=args.UCR_folder)
    elif args.dataset == 'txt_all':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_txt_all(path, folder=args.UCR_folder)

args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))
args.lr_decay_steps = args.eval_per_steps

if args.device is None:
    if torch.backends.mps.is_available():
        args.device = 'mps'
    elif torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
print(f"Using device: {args.device}")

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
