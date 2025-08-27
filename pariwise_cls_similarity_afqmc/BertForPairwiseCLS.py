import torch
from torch import nn
from transformers import AutoModel
from AFQMCDataLoader import checkpoint

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        super(BertForPairwiseCLS, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vector = bert_output.last_hidden_state[:, 0, :]
        cls_vector = self.dropout(cls_vector)
        logits = self.classifier(cls_vector)
        return logits


model = BertForPairwiseCLS().to(device)
print(model)
