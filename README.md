
# AI Medical Assistant Chatbot - Fine-tuning and Testing

## Overview

This project involves fine-tuning a pretrained language model using the Unsloth framework for building an AI Medical Assistant Chatbot. The chatbot is designed to answer medical questions based on a specific dataset.

## Setup

### Install Required Packages

The following commands will install all necessary packages, including Unsloth, Xformers (Flash Attention), and other dependencies:

```python
# Install Unsloth, Xformers (Flash Attention), and other packages
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes"
!pip install triton
!pip uninstall xformers
!pip install xformers
```

### Import Modules

After installing the necessary packages, import the required modules:

```python
from unsloth import FastLanguageModel
import torch
```

## Model Loading and Configuration

### Model Configuration

The model is configured with the following parameters:

- **max_seq_length**: The maximum sequence length for the model (e.g., 2048).
- **dtype**: Data type (e.g., `None` for auto-detection, `float16` for certain GPUs).
- **load_in_4bit**: Whether to use 4-bit quantization to reduce memory usage.

### Pretrained Models

A list of pre-quantized 4-bit models supported by Unsloth is provided:

```python
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
]
```

### Loading and Configuring the Model

Load the model using the specified configuration:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

Configure the model with PEFT (Parameter-Efficient Fine-Tuning):

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

## Dataset Preparation

### Loading the Dataset

The medical dataset is loaded using the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("ruslanmv/ai-medical-dataset")

if 'train' in dataset:
    dataset = dataset['train'].select(range(500))
```

### Formatting the Dataset

The dataset is formatted to match the chatbot's expected input structure:

```python
medical_prompt = """You are an AI Medical Assistant Chatbot, trained to answer medical questions. Below is an instruction that describes a task, paired with an response context. Write a response that appropriately completes the request.

### Question:
{}


### Context:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    questions = examples["question"]
    contexts = examples["context"]
    texts = []
    for question, context in zip(questions, contexts):
        text = medical_prompt.format(question, context) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

## Training

### Trainer Configuration

Configure the trainer using the `SFTTrainer` and `TrainingArguments`:

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
```

### Training Execution

Run the training:

```python
trainer_stats = trainer.train()
```

## Inference and Testing

### Testing Before Training

To test the model before training, use the following script:

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [medical_prompt.format("What is the resurgent sodium current in mouse cerebellar Purkinje neurons?", "")],
    return_tensors="pt"
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
```

### Testing After Training

After training, test the model again to evaluate the performance:

```python
if False:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [medical_prompt.format("What is the resurgent sodium current in mouse cerebellar Purkinje neurons?", "") + tokenizer.eos_token],
    return_tensors="pt"
).to("cuda")

model.eval()

_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
```

## Model Saving

Finally, save the trained model and tokenizer:

```python
model.push_to_hub("HadeelHegazi/ai_medical_dataset_train_epochs1max_steps60", token="your_huggingface_token")
tokenizer.push_to_hub("HadeelHegazi/ai_medical_dataset_train_epochs1max_steps60", token="your_huggingface_token")
```

## Issues and Debugging

If you encounter any issues, such as those depicted in the screenshot, ensure that you:

1. **Check for Correct GPU Usage:** Ensure your code is running on the correct GPU.
2. **Clear CUDA Cache:** Use `torch.cuda.empty_cache()` and `gc.collect()` to clear the cache.
3. **Check Model and Tokenizer Configuration:** Ensure all configurations are correct for the specific model.








---
base_model: unsloth/meta-llama-3.1-8b-bnb-4bit
language:
- en
license: apache-2.0
tags:
- text-generation-inference
- transformers
- unsloth
- llama
- trl
---

# Uploaded  model

- **Developed by:** HadeelHegazi
- **License:** apache-2.0
- **Finetuned from model :** unsloth/meta-llama-3.1-8b-bnb-4bit

This llama model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
