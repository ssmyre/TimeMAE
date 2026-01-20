import numpy as np
import torch
import torch.utils.data as Data


# class Dataset(Data.Dataset):
#     def __init__(self, device, mode, data, wave_len):
#         self.device = device
#         self.datas, self.label = data
#         self.mode = mode
#         self.wave_len = wave_len
#         self.__padding__()

#     def __padding__(self):
#         origin_len = self.datas[0].shape[0]
#         if origin_len % self.wave_len:
#             padding_len = self.wave_len - (origin_len % self.wave_len)
#             padding = np.zeros((len(self.datas), padding_len, self.datas[0].shape[1]), dtype=np.float32)
#             self.datas = np.concatenate([self.datas, padding], axis=-2)

#     def __len__(self):
#         return len(self.datas)

#     def __getitem__(self, item):
#         data = torch.tensor(self.datas[item]).to(self.device)
#         label = self.label[item]
#         return data, torch.tensor(label).to(self.device)

#     def shape(self):
#         return self.datas[0].shape
class Dataset(Data.Dataset):
    def __init__(self, device, mode, data, target, wave_len, analysis = 0):
        # self.device = device
        self.datas = torch.tensor(data[0].astype(np.float32))  # shape: [N, T]
        self.Neurotrans = torch.tensor(data[1].astype(np.float32))  # shape: [N, 2] [dopamine, serotonin]
        self.Electrode = data[2]
        self.target = target
        if self.target == 'dopamine':
            self.Neurotrans = self.Neurotrans[:,0].unsqueeze(1)
        elif self.target == 'serotonin':
            self.Neurotrans = self.Neurotrans[:,1].unsqueeze(1)
        else:
            pass
        self.mode = mode
        self.wave_len = wave_len
        self.analysis = analysis
        self.__padding__()

    def __padding__(self):
        origin_len = self.datas[0].shape[0]
        if origin_len % self.wave_len:
            padding_len = self.wave_len - (origin_len % self.wave_len)
            padding = torch.zeros((len(self.datas), padding_len, self.datas[0].shape[1]), dtype=torch.float32)
            self.datas = torch.concatenate([self.datas, padding], axis=1)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        if self.analysis == 1:
            return self.datas[item], self.Neurotrans[item], self.Electrode[item]
        else:
            return self.datas[item], self.Neurotrans[item]

    def shape(self):
        return self.datas[0].shape
