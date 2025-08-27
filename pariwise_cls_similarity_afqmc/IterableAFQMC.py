from torch.utils.data import IterableDataset
import json


class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample


train_data = IterableAFQMC('/Users/hm/mycode/transformer/data/afqmc_public/train.json')

print(next(iter(train_data)))
