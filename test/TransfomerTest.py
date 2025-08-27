import torch
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
result = classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
                    )
print(result)

classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library",
                    candidate_labels=["education", "politics", "business"])
print(result)

generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
print(result)
result = generator("In this course, we will teach you how to", num_return_sequences=2, max_length=50)
print(result)

generator = pipeline("text-generation", model="distilgpt2")
result = generator("in this course, we will teach you how to",
                   max_length=30,
                   num_return_sequences=2)
print(result)

generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
result = generator("[CLS] 万 叠 春 山 积 雨 晴 ，",
                   max_length=40,
                   num_return_sequences=2
                   )
print(result)

unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(result)

ner = pipeline("ner", grouped_entities=True)
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)

question_answer = pipeline("question-answering")
answer = question_answer(question="Where do I work?", context="My name is Sylvain and I work at Hugging Face in "
                                                              "Brooklyn", )
print(answer)

summary = pipeline("summarization")
result = summary(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
    """
)
print(result)

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]
inputs = tokenizer(raw_inputs, return_tensors="pt", padding=True, truncation=True)
print(inputs)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
inputs = tokenizer(raw_inputs, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
print(outputs.logits.shape)
print(outputs.logits)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)