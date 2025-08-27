from torch.utils.data import Dataset
import json


class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = AFQMC('/Users/hm/mycode/transformer/data/afqmc_public/train.json')
valid_date = AFQMC('/Users/hm/mycode/transformer/data/afqmc_public/dev.json')

print(train_data[0])
