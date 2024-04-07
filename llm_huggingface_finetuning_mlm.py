# -*- coding: utf-8 -*-
"""LLM_HuggingFace_Finetuning_MLM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DKF6L0fedPFFdL93YGcI3PP2SFYbVbMj
"""

#!pip install datasets evaluate transformers[sentencepiece]
!pip install datasets
!pip install transformers[torch]
!pip install accelerate -U
# To run the training on TPU, you will need to uncomment the following line:
# !pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
# !apt install git-lfs

from huggingface_hub import notebook_login

notebook_login()

"""**Overall objective - take a wiki trained distilBERT model and fine it using imdb data for text generation**"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
# this is 67M parameter model and much smaller than BERT
# trained on wikipedia and bookcorpus datasets
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# sample text to evaluate the current training of distilbert
text = "This is a great [MASK]."

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# identify mask location and it's logits
# mask location
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# mask logits
mask_token_logits = token_logits[0, mask_token_index, :]
# pick the mask candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
# printing the masked value
for token in top_5_tokens:
  print(f""" >>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))} """)

"""**Fine tuning distilBERT on imdb data**"""

from datasets import load_dataset
imdb_dataset = load_dataset("imdb")
sample = imdb_dataset["unsupervised"].shuffle(seed=42).select(range(3))
for row in sample:
  print(f"\n' >>> Review: {row['text']}'")
  print(f"\n' >>> Label: {row['label']}'")

"""## In order to prevent loss of information via truncation - combine all the examples together
#### step_1 - creating tokens using tokenizer
#### step_2 - combine examples and split
#### step_3 - to add masks randomly - mask few words randomly in a given sentence / mask > 1 word in a given sentence
"""

def tokenize_function(examples):
    # tokenizer returns input_id, attention_mask
    # in addition to this - add word_ids
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
tokenized_datasets

def group_texts(examples):
    # size picked based on GPU memory
    chunk_size = 128
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets

# fine tuning model for text generation is similar as that of fine tuning a model for classification purpose
# except masks have to be inserted for the model to understand what is mask
from transformers import DataCollatorForLanguageModeling
# randomly masks words in the sample data provided
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    # for each sentence within the sample
    # create map between word and token idx
    # randomly select an idx from the map and mask the word
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset

from huggingface_hub import notebook_login

notebook_login()

from transformers import TrainingArguments

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs = 2,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=False,
    logging_steps=logging_steps,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
# perplexity = e^(cross entropy loss)
print(f"""Perplexity : {math.exp(trainer.evaluate())} """)

from huggingface_hub import notebook_login

notebook_login()

from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name

# alternative approach to train the model using accelerator - skipping this for now
# model inference using trained model
from transformers import pipeline
# load the model from the file path where it is written to
# default file path - sankarigovindarajan/distilbert-finetuned-imdb
mask_filler = pipeline(
    "fill-mask", model="sgovindarajan/distilbert-base-uncased-finetuned-imdb"
)
preds = mask_filler(text)
for pred in preds:
    print(f">>> {pred['sequence']}")

!apt install git-lfs



"""**Fine tuning the model by LORA**"""