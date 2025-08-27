import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer
from math import sqrt
import torch.nn.functional as F

model_cjpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_cjpt)

text = "time flies like an arrow"
inputs = tokenizer(text,return_tensors="pt",add_special_tokens=False)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_cjpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())

# tensor([[2051,10029,2066,2019,8612]])
# Embedding(30522,768)
# torch.Size([1,5,768])

Q = K = V = inputs_embeds
dim_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(dim_k)
print(scores.size())

weight = F.softmax(scores, dim=-1)
print(weight.sum(dim=-1))

attn_output = torch.bmm(weight, V)
print(attn_output.shape)

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None :
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weight = F.softmax(scores, dim=-1)
    return torch.bmm(weight, value)

