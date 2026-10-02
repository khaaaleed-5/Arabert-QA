---
title: AraBERT Arabic QA
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
models:
- Khaaaleed5/arabert-qa
---

# AraBERT Arabic Question Answering

Fine-tuning [AraBERT v0.2](https://huggingface.co/aubmindlab/bert-base-arabertv02) for extractive Arabic question answering using the [ArabicaQA](https://huggingface.co/datasets/abdoelsayed/ArabicaQA) dataset.

The resulting model is available on Hugging Face:

- **Model:** [Khaaaleed5/arabert-qa](https://huggingface.co/Khaaaleed5/arabert-qa)
- **Interactive demo:** [AraBERT Arabic QA Space](https://huggingface.co/spaces/Khaaaleed5/arabert-qa)

## Overview

Fine-tuning AraBERT to find an answer span inside an Arabic context. Given a question and a passage, the model predicts the start and end positions of the answer in that passage.

It includes:

- Dataset loading and preparation from ArabicaQA
- Answer-offset validation and removal of misaligned examples
- A context-level train/validation split to reduce data leakage
- Long-context tokenization with overlapping windows
- A Gradio interface for local or Hugging Face deployment

## Repository Structure

```text
.
├── app.py                  # Gradio inference application
├── finetuneArabert.ipynb   # Dataset preparation and fine-tuning workflow
├── requirements.txt        # Dependencies for the demo application
└── README.md
```

## Model Details

| Item | Value |
| --- | --- |
| Base model | `aubmindlab/bert-base-arabertv02` |
| Task | Extractive question answering |
| Dataset | `abdoelsayed/ArabicaQA` |
| Maximum sequence length | 384 tokens |
| Document stride | 128 tokens |
| Training epochs | 3 |
| Learning rate | `3e-5` |
| Train/evaluation batch size | 8 per device |
| Weight decay | `0.01` |
| Evaluation | At the end of each epoch |



## Try the model in the Hugging Face Space: [AraBERT Arabic QA](https://huggingface.co/spaces/Khaaaleed5/arabert-qa)

## Inference Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_id = "Khaaaleed5/arabert-qa"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_id)

question = "ما هي عاصمة مصر؟"
context = "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها."

inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=384)
with torch.no_grad():
    outputs = model(**inputs)

start = outputs.start_logits.argmax()
end = outputs.end_logits.argmax() + 1
answer = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
print(answer)
```

Expected answer:

```text
القاهرة هي عاصمة جمهورية مصر العربية
```

## Limitations

- This is an extractive QA model: answers must be spans present in the supplied context.
- It may return incorrect or empty answers when the context does not contain the answer.

