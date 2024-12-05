pip install transformers datasets torch rouge-score kagglehub

pip install tqdm

import torch
import torch.nn as nn
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, logging as transformers_logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import time

transformers_logging.set_verbosity_error()

# Step 1: Download dataset from Kaggle
path = "/kaggle/input/cnn-dailymail"
data_files = {
    "train": os.path.join(path, "train.csv"),
    "validation": os.path.join(path, "validation.csv"),
    "test": os.path.join(path, "test.csv")
}
dataset = load_dataset("csv", data_files=data_files)


# Sampling 10% of the train data
train_size = len(dataset['train'])
subset_size = int(0.1 * train_size)
indices = random.sample(range(train_size), subset_size)
train_subset = Subset(dataset['train'], indices)

# Load model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define hyperparameters
batch_size = 32
learning_rate = 5e-5
num_epochs = 1
max_input_length = 128
max_output_length = 64

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Freeze GPT-2 model parameters
for param in model.parameters():
    param.requires_grad = False

# Soft Prompt Embedding Layer
num_soft_tokens = 5
soft_embedding_layer = nn.Embedding(num_soft_tokens, model.config.n_embd).to(device)
soft_embedding_layer.weight.requires_grad = True

# Only the soft embeddings are trainable
trainable_params = sum(p.numel() for p in soft_embedding_layer.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

# Preprocessing inside the DataLoader using a collate function
def collate_fn(batch):
    articles = ["[SUMMARIZE] " + example['article'] for example in batch]
    summaries = [example['highlights'] for example in batch]

    # Tokenize inputs and labels
    inputs = tokenizer(articles, max_length=max_input_length, truncation=True, padding='max_length', return_tensors="pt")
    labels = tokenizer(summaries, max_length=max_output_length, truncation=True, padding='max_length', return_tensors="pt").input_ids

    # Pad labels to match input length
    labels_padded = torch.full((labels.size(0), max_input_length + num_soft_tokens), -100, dtype=torch.long)  # Adjust label padding length
    labels_padded[:, num_soft_tokens:num_soft_tokens + labels.size(1)] = labels  # Shift labels to match the extended input length

    inputs['labels'] = labels_padded
    return inputs

# Create DataLoader
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize optimizer for soft embeddings only
optimizer = AdamW(soft_embedding_layer.parameters(), lr=learning_rate)

# Move model and soft embedding layer to the device
model.to(device)
soft_embedding_layer.to(device)

print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

start_time = time.time()
# Fine-tuning loop with progress bar
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    print(f"\nStarting epoch {epoch + 1}/{num_epochs}")

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", dynamic_ncols=True)

    for batch in progress_bar:
        optimizer.zero_grad()

        # Move inputs and labels to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Concatenate soft prompt embeddings with input embeddings
        current_batch_size = input_ids.size(0)
        soft_embeddings = soft_embedding_layer.weight.unsqueeze(0).expand(current_batch_size, -1, -1)
        input_embeddings = model.transformer.wte(input_ids)
        input_embeddings = torch.cat([soft_embeddings, input_embeddings], dim=1)

        # Adjust attention mask for soft tokens
        attention_mask = torch.cat([torch.ones(current_batch_size, num_soft_tokens).to(device), attention_mask], dim=1)

        # Forward pass
        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining complete! Total training time: {training_time / 60:.2f} minutes.")

# Save the model's weights as a .pt file after training
torch.save(model.state_dict(), "./gpt-prompttune-weights.pt")
print("Model weights saved as gpt-prompttune-weights.pt")

# Move model to the appropriate device (GPU if available, otherwise CPU)
model.to(device)

# Evaluation - Calculate ROUGE scores on a few samples
def collate_fn_eval(batch):
    articles = ["[SUMMARIZE] " + example['article'] for example in batch]
    summaries = [example['highlights'] for example in batch]

    inputs = tokenizer(articles, max_length=max_input_length, truncation=True, padding='max_length', return_tensors="pt")
    
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'summaries': summaries}

# Create DataLoader for evaluation
test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn_eval)

# Define ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Evaluation loop
model.eval()
predictions, references = [], []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Create soft embeddings and concatenate with input embeddings
        soft_embeddings = soft_embedding_layer.weight.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        input_embeddings = model.transformer.wte(input_ids)
        input_embeddings = torch.cat([soft_embeddings, input_embeddings], dim=1)

        # Adjust attention mask for the soft tokens
        attention_mask = torch.cat([torch.ones(input_ids.size(0), num_soft_tokens).to(device), attention_mask], dim=1)

        # Generate predictions with modified embeddings
        generated_tokens = model.generate(inputs_embeds=input_embeddings, attention_mask=attention_mask, max_new_tokens=max_output_length, num_beams=5)

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = batch['summaries']  # Original summaries

        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

    
# Calculate ROUGE scores
rouge_scores = []
for pred, ref in zip(predictions, references):
    score = scorer.score(pred, ref)
    rouge_scores.append(score)

# Calculate average scores
avg_rouge1 = sum([s['rouge1'].fmeasure for s in rouge_scores]) / len(rouge_scores)
avg_rouge2 = sum([s['rouge2'].fmeasure for s in rouge_scores]) / len(rouge_scores)
avg_rougeL = sum([s['rougeL'].fmeasure for s in rouge_scores]) / len(rouge_scores)

print(f"Average ROUGE-1: {avg_rouge1}")
print(f"Average ROUGE-2: {avg_rouge2}")
print(f"Average ROUGE-L: {avg_rougeL}")



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


pip install transformers datasets peft evaluate

import os
import numpy as np
import random
from datasets import load_dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, logging as transformers_logging
import torch
from peft import get_peft_model, LoraConfig, TaskType
import kagglehub
import evaluate
from torch.utils.data import DataLoader, Subset
import time

transformers_logging.set_verbosity_error()

path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
print("Path to dataset files:", path)

data_files = {
    "train": os.path.join(path, 'cnn_dailymail', "train.csv"),
    "validation": os.path.join(path, 'cnn_dailymail', "validation.csv"),
    "test": os.path.join(path, 'cnn_dailymail', "test.csv")
}
dataset = load_dataset("csv", data_files=data_files)

# Sampling 10% of the train data
train_size = len(dataset['train'])
subset_size = int(0.1 * train_size)
indices = random.sample(range(train_size), subset_size)
train_subset = Subset(dataset['train'], indices)

# Step 3: Initialize tokenizer
print("\nInitializing tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized. Pad token set to:", tokenizer.pad_token)

# Custom collate function for training
def custom_collate_fn(batch):
    
    inputs = [example['article'] for example in batch]  
    targets = [example['highlights'] for example in batch]  
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length', return_tensors='pt')  
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print("\nSetting up custom data collator.")
data_collator = custom_collate_fn

# Step 5: Initialize GPT-Neo model
print("\nLoading GPT-Neo 125M model...")
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
model.config.pad_token_id = tokenizer.eos_token_id
print("Model loaded. Pad token ID set to:", model.config.pad_token_id)

# Applying LoRA
print("\nApplying LoRA configuration...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
print("LoRA applied successfully.")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"\nUsing device: {device}")
model.to(device)

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of trainable parameters: {trainable_params}")

# Training arguments with `remove_unused_columns=False`
print("\nSetting up training arguments...")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to='none',
    remove_unused_columns=False 
)
print("Training arguments set.")

print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Trainer initialized.")

start_time = time.time()

print("\nStarting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")

end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining complete! Total training time: {training_time / 60:.2f} minutes.")

# Save the model's weights as a .pt file after training
torch.save(model.state_dict(), "./gpt-neo-Lora-weights.pt")
print("Model weights saved as gpt-neo-Lora-weights.pt")

# Evaluation on test set
print("\nEvaluating the model on the test set...")

# Custom DataLoader for the test set
print("\nSetting up DataLoader for the test set...")
test_dataloader = DataLoader(dataset['test'], batch_size=8, collate_fn=lambda x: x)

# Function to generate summaries in batches
def generate_summaries(model, dataloader):
    model.eval()
    generated_summaries = []
    references = []
    
    print("\nGenerating summaries...")
    for batch_idx, batch in enumerate(dataloader):
        texts = [example['article'] for example in batch]  
        refs = [example['highlights'] for example in batch]  
        references.extend(refs)       
        inputs = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding=True)
        input_ids = inputs['input_ids'].to(device) 
        attention_mask = inputs['attention_mask'].to(device)     
        # Generate summaries
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,  
                num_beams=5,
                early_stopping=True
            )
        # Decode summaries
        for output in outputs:
            summary = tokenizer.decode(output, skip_special_tokens=True)
            generated_summaries.append(summary)
    
    return generated_summaries, references

# Generate summaries
generated_summaries, references = generate_summaries(model, test_dataloader)
print("Summaries generated successfully.")

# Computing ROUGE scores
print("\nComputing ROUGE scores...")
rouge_metric = evaluate.load('rouge')
rouge_scores = rouge_metric.compute(predictions=generated_summaries, references=references)
print("ROUGE scores computed:")
print(rouge_scores)

# Evaluating loss on the test set
print("\nComputing loss on the test set...")
test_loss = trainer.evaluate(eval_dataset=dataset['test'])
print("Test set evaluation complete.")
print("Test Loss:", test_loss['eval_loss'])


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import random
from datasets import load_dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AdamW, get_scheduler
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import evaluate
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


path = "/kaggle/input/cnn-dailymail"
#print("Path to dataset files:", path)

# Load dataset
data_files = {
    "train": os.path.join(path, "train.csv"),
    "validation": os.path.join(path, "validation.csv"),
    "test": os.path.join(path, "test.csv")
}
dataset = load_dataset("csv", data_files=data_files)

input_column = 'article'      
target_column = 'highlights'  

# Step 2: Sample 10% of the training data
train_size = len(dataset['train'])
subset_size = int(0.1 * train_size)
print(f"\nTotal number of training examples: {train_size}")
print(f"Sampling 1% of the training data: {subset_size} examples")
indices = random.sample(range(train_size), subset_size)
train_subset = Subset(dataset['train'], indices)
print("Train subset created.")

# Step 3: Initialize tokenizer
print("\nInitializing tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized. Pad token set to:", tokenizer.pad_token)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Step 4: Define custom collate function
def custom_collate_fn(batch):
    inputs = [example[input_column] for example in batch]
    targets = [example[target_column] for example in batch]

    # Tokenize inputs and targets separately
    tokenized_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    tokenized_targets = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Concatenate inputs and targets for conditional generation
    input_ids = tokenized_inputs['input_ids']
    target_ids = tokenized_targets['input_ids']

    # Create labels: mask the input tokens with -100 and set labels for target tokens
    labels = torch.cat([
        torch.full((input_ids.size(0), input_ids.size(1)), -100, dtype=torch.long),
        target_ids
    ], dim=1)

    # Concatenate input_ids and target_ids
    input_ids = torch.cat([input_ids, target_ids], dim=1)

    # Update attention_mask
    attention_mask = torch.cat([
        tokenized_inputs['attention_mask'],
        tokenized_targets['attention_mask']
    ], dim=1)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Step 5: Initialize GPT-Neo model
print("\nLoading GPT-Neo 125M model...")
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
model.config.pad_token_id = tokenizer.eos_token_id

# Step 6: Freeze all layers except the last transformer block
print("\nFreezing all layers except the last transformer block...")
for name, param in model.named_parameters():
    if 'transformer.h.' in name:
        layer_num = int(name.split('.')[2])
        if layer_num != (model.transformer.h.__len__() - 1): 
            param.requires_grad = False
    else:
        param.requires_grad = True  
print("Freezing complete. Only the last transformer block is trainable.")

# Step 7: Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of trainable parameters: {trainable_params}")
    
model.to(device)
# Step 8: Check GPU availability and print GPU information
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
# Step 9: Set up DataLoader
print("\nSetting up DataLoader...")
train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
print("DataLoader set up successfully.")

# Step 10: Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) 
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_training_steps=num_training_steps
)


print("\nStarting training...")
start_time = time.time()
num_epochs = 10
max_grad_norm = 1.0

for epoch in range(num_epochs):
    model.train()  
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{num_epochs} finished.")


end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining complete! Total training time: {training_time / 60:.2f} minutes.")

# Step 12: Save the fine-tuned model
print("\nSaving the fine-tuned model...")
model.save_pretrained("./gpt-neo-finetuned")
tokenizer.save_pretrained("./gpt-neo-finetuned")
print("Model saved successfully.")

# Save the model's weights as a .pt file after training
torch.save(model.state_dict(), "./gpt-neo-finetuned-weights.pt")
print("Model weights saved as gpt-neo-finetuned-weights.pt")

test_dataloader = DataLoader(dataset['test'], batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
# Step 13: Evaluate the model on the test set (test loss)
print("\nEvaluating the model on the test set for loss...")
model.eval() 
total_loss = 0
num_batches = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1

avg_test_loss = total_loss / num_batches
print(f"Average Test Loss: {avg_test_loss}")

# Step 14: Generate summaries and compute ROUGE scores
print("\nGenerating summaries and computing ROUGE scores...")
rouge_metric = evaluate.load('rouge')

generated_summaries = []
references = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=128,
            num_beams=5,
            early_stopping=True
        )

        # Decode generated summaries
        decoded_summaries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_summaries.extend(decoded_summaries)

        # Add references (ground truth summaries)
        decoded_references = [
            tokenizer.decode([token_id for token_id in target if token_id != -100], skip_special_tokens=True) 
            for target in batch['labels']
        ]
        references.extend(decoded_references)

# Compute ROUGE scores
rouge_scores = rouge_metric.compute(predictions=generated_summaries, references=references)
print(f"\nROUGE Scores: {rouge_scores}")
