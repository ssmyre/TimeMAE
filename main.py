# import torch
# import warnings

# warnings.filterwarnings('ignore')
# from args import args, Test_data, Valid_data, Train_data
# from dataset import Dataset
# from model.TimeMAE import TimeMAE
# from process import Trainer_Regress
# import torch.utils.data as Data
# import torch.nn as nn
# # import cProfile

# def main():
#     torch.set_num_threads(12)
#     torch.cuda.manual_seed(3407)
#     train_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data, wave_len=args.wave_length)
#     train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
#     args.data_shape = train_dataset.shape()
#     valid_dataset = Dataset(device=args.device, mode='validate', data=Valid_data, wave_len=args.wave_length)
#     valid_loader = Data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True)
#     test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
#     test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

#     print(args.data_shape)
#     print('dataset initial ends')

#     model = TimeMAE(args)
#     # model = nn.DataParallel(model)
#     print('model initial ends')
#     trainer = Trainer_Regress(args, model, train_loader, valid_loader, test_loader, verbose=True)
#     if args.resume == 0:
#         trainer.pretrain()
#     trainer.finetune()


# if __name__ == '__main__':
#     main()

import torch
import warnings

warnings.filterwarnings('ignore')
from args import args, Test_data, Train_data_all, Train_data
from dataset import Dataset
from model.TimeMAE import TimeMAE
from processv2 import Trainer_Regress
import torch.utils.data as Data
import torch.nn as nn
import cProfile

def main():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all, target=args.target, wave_len=args.wave_length)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    train_finetune_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data, target=args.target, wave_len=args.wave_length)
    train_finetune_loader = Data.DataLoader(train_finetune_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data, target=args.target, wave_len=args.wave_length)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    print(args.data_shape)
    print('dataset initial ends')

    model = TimeMAE(args)
    # model = nn.DataParallel(model)
    print('model initial ends')
    trainer = Trainer_Regress(args, model, train_loader, train_finetune_loader, test_loader, verbose=True)
    if args.resume == 0:
        trainer.pretrain()
    trainer.finetune()


if __name__ == '__main__':
    main()
