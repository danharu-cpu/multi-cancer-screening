import os
import torch
import torch.utils.data as data
import datatable as dt

def default_loader(path):
    datatable_df = dt.fread(path, sep=',', header=None)
    return datatable_df.to_pandas()

class CustomDataset(data.Dataset):
    def __init__(self, data_sens, gas_label, mode='hdcnn', loader=default_loader):
        self.mode = mode
        self.data_sens = loader(data_sens).values
        self.gas_label = loader(gas_label).values

    def __getitem__(self, index):
        data_sens = torch.FloatTensor(self.data_sens[index])

        if self.mode == 'hdcnn':
            coarse_label = torch.tensor(self.gas_label[index][0], dtype=torch.long)
            fine_label = torch.tensor(self.gas_label[index][1], dtype=torch.long)
            return data_sens, coarse_label, fine_label

        elif self.mode == 'cnn':
            final_label = torch.tensor(self.gas_label[index][0], dtype=torch.long)
            return data_sens, final_label

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __len__(self):
        return len(self.data_sens)
