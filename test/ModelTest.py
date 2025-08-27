from transformers import AutoModel, BertTokenizer
from transformers import AutoTokenizer

# 正常都是使用AutoModel模型这样不管使用那种模型，只需要切换checkpoint就可以了
model = AutoModel.from_pretrained('bert-base-cased')
model.save_pretrained('/Users/hm/mycode/transformers/model/bert-base-cased')
# 同样的我们使用AutoTokenizer的分词加载器
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
tokenizer.save_pretrained('/Users/hm/mycode/transformers/model/bert-base-cased')

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

# ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
# Bert模型采用的子词分词器
print(tokens)

# 编码
# [7993, 170, 13809, 23763, 2443, 1110, 3014]
# 获取到每个token对应的词典的id编号
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102]
# encode增加了头尾的编号[CLS]和[SEP]
sequence_ids = tokenizer.encode(sequence)
print(sequence_ids)

# {'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
# 正常操作，获取到正常的input_ids和类型跟掩码
tokenizer_text = tokenizer(sequence)
print(tokenizer_text)

# 解码
# 将编码后的token进行解码，获取到我们所熟知语句
# Using a Transformer network is simple
decoded = tokenizer.decode(ids)
print(decoded)

# [CLS] Using a Transformer network is simple [SEP]
decoded = tokenizer.decode(sequence_ids)
print(decoded)

import torch
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# tokens = tokenizer.tokenize(sequence)
# 实际场景中不需要转换成ids，而这样的转换也使得distilbert分词器中必须带有的[CLS][SEP]省略会产生一定的误解，而
# model之所以没有报错则是因为自动填充了
# ids = tokenizer.convert_tokens_to_ids(tokens)
# 正确写法
tokens = tokenizer(sequence, return_tensors="pt")
print("Input Keys:\n", tokens.keys())
print("\nInput IDs:\n", tokens["input_ids"])

# input_ids = torch.tensor([ids])
# print("Input IDs:\n", input_ids)

# outputs = model(input_ids)
outputs = model(**tokens)
print("Logits:\n", outputs.logits)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batch_id = [[200, 200, 200],
            [200, 200, tokenizer.pad_token_id],
            ]
batch_attention_mask = [[1, 1, 1],
                        [1, 1, 0],
                        ]
print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
outputs = model(torch.tensor(batch_id), attention_mask=torch.tensor(batch_attention_mask))
print(outputs.logits)

sequences = ["I've been waiting for a HuggingFace course my whole life",
             "So have I!"]
model_inputs = tokenizer(sequences)
# {'input_ids': [[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 102],
#               [101, 2061, 2031, 1045, 999, 102]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1]]}
# 需要使用padding进行填充
print(model_inputs)

# {'input_ids': [[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 102],
#               [101, 2061, 2031, 1045, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#   'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
model_inputs = tokenizer(sequences, padding="longest")
print(model_inputs)
# padding:longest为当前语句中最长的语句的补充
#         max_length为当前的能够接受的最长数据，该模型为512
model_inputs = tokenizer(sequences, padding="max_length")
print(model_inputs)

# 截断操作使用truncation：当模型超过512时会自动截断，或者使用max_length来进行截断限制
# {'input_ids': [[101, 1045, 1005, 2310, 2042, 3403, 2005, 102],
#                [101, 2061, 2031, 1045, 999, 102]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1]]}
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
print(model_inputs)

# 分词器还可以通过 return_tensors 参数指定返回的张量格式：
# 设为 pt 则返回 PyTorch 张量；
#     tf 则返回 TensorFlow 张量，
#     np 则返回 NumPy 数组。

# {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,
# 2166,   102], [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0,
# 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
# 0, 0, 0]])}
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
print(model_inputs)

# model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
# print(model_inputs)

# {'input_ids': array([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,12172,  2607,  2026,  2878,
# 2166,   102], [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,    0,     0,     0,     0,     0,
# 0]]), 'attention_mask': array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
# 0, 0, 0]])}
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
print(model_inputs)

# 正常使用方式
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
outputs = model(**tokens)
# 逻辑值不代表概率值
print(outputs.logits.shape)

# 概率值计算
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(tokens)

sentence1_list = ["First sentence.", "This is the second sentence.", "Third one."]
sentence2_list = ["First sentence is short.", "The second sentence is very very very long.", "ok."]

tokens = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(tokens)
print(tokens["input_ids"].shape)

sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'
print(tokenizer.tokenize(sentence))

num_added_tokens = tokenizer.add_tokens(["new_token1", "my_new-token2"])
print("We have added", num_added_tokens, "tokens")

new_tokens = ["new_token1", "my_new-token2"]
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
tokenizer.add_tokens(list(new_tokens))

special_tokens_dict = {"cls_token": "[MY_CLS]"}

num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_tokens, "tokens")

assert tokenizer.cls_token == "[MY_CLS]"

num_added_toks = tokenizer.add_tokens(["[NEW_tok1]", "[NEW_tok2]"])
num_added_toks = tokenizer.add_tokens(["[NEW_tok3]", "[NEW_tok4]"], special_tokens=True)

print("We have added", num_added_toks, "tokens")
print(tokenizer.tokenize('[NEW_tok1] Hello [NEW_tok2] [NEW_tok3] world [NEW_tok4]!'))

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)
print("We have added", num_added_toks, "tokens")

sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'
print(tokenizer.tokenize(sentence))

model.resize_token_embeddings(len(tokenizer))
print(model)
print(model.embeddings.word_embeddings.weight.size())

print(model.embeddings.word_embeddings.weight[-2:, :])

with torch.no_grad():
    model.embeddings.word_embeddings.weight[-2:, :] = torch.zeros([2, model.config.hidden_size], requires_grad=True)
print(model.embeddings.word_embeddings.weight[-2:, :])

token_id = tokenizer.convert_tokens_to_ids('entity')
token_embeddings = model.embeddings.word_embeddings.weight[token_id]
print(token_id)

with torch.no_grad():
    for i in range(1, num_added_toks + 1):
        model.embeddings.word_embeddings.weight[-i:, :] = token_embeddings.clone().detach().requires_grad_(True)
print(model.embeddings.word_embeddings.weight[-2:, :])

descriptions = ['start of entity', 'end of entity']

with torch.no_grad():
    for i, token in enumerate(reversed(descriptions), start=1):
        tokenized = tokenizer.tokenize(token)
        print(tokenized)
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
        new_embeddings = model.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
        model.embeddings.word_embeddings.weight[-i:, :] = new_embeddings.clone().detach().requires_grad_(True)
print(model.embeddings.word_embeddings.weight[-2::])
