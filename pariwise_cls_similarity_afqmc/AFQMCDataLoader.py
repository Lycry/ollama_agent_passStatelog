import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from AFQMC import train_data

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def collate_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y


train_dataLoader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)

batch_X, batch_y = next(iter(train_dataLoader))
print('batch_X.shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y.shape:', batch_y.shape)
print(batch_X)
print(batch_y)
