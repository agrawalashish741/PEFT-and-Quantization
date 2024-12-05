# Install required libraries
!pip install transformers bitsandbytes datasets psutil

pip install transformers datasets bitsandbytes psutil pandas

pip install -U bitsandbytes

pip install -U accelerate

# !pip install transformers datasets psutil

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
from torch.utils.data import DataLoader
import psutil

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the model name
model_name = 'gpt2'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load the dataset
dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
data = dataset['sentence'][:3000]  # Get the first 3000 sentences

# Function to compute perplexity
def compute_perplexity(model, tokenizer, texts, batch_size=8):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    dataloader = DataLoader(texts, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Exclude padding tokens from loss calculation
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Scale loss by the number of tokens
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# Function to measure inference latency
def measure_latency(model, tokenizer, texts, batch_size=8):
    model.eval()
    total_time = 0.0
    total_sentences = 0
    dataloader = DataLoader(texts, batch_size=batch_size)
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
            start_time = time.time()
            outputs = model(**inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
            total_sentences += len(batch)
    avg_latency = total_time / total_sentences
    return avg_latency

# Function to measure memory usage
def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / (1024 ** 2)  # in MB


# Function to quantize weights to INT8
def quantize_weights(weights):
    min_val = weights.min()
    max_val = weights.max()
    scale = (max_val - min_val) / 255
    scale = max(scale, 1e-8)  # Prevent division by zero
    zero_point = min_val
    quantized = ((weights - min_val) / scale).round().clamp(0, 255).to(torch.uint8)
    return quantized, scale, zero_point

# Function to dequantize weights back to FP32
def dequantize_weights(quantized, scale, zero_point):
    return quantized.float() * scale + zero_point

# Custom INT8 Linear Layer
class QuantizedLinear(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Quantize weights per layer
        self.weight_int8, self.scale, self.zero_point = quantize_weights(original_layer.weight.data)

        # Keep bias in FP32
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, input):
        # Dequantize weights for computation
        weight_fp32 = dequantize_weights(self.weight_int8, self.scale, self.zero_point).to(input.device)
        output = torch.matmul(input, weight_fp32.t())
        if self.bias is not None:
            output += self.bias
        return output

# Function to quantize the entire model
def quantize_model_manually(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinear(module))
        else:
            quantize_model_manually(module)
    return model

# Selective Component Quantization
def selective_quantization(model, components_to_quantize):
    """
    Quantizes only the specified components of the model.
    Args:
        model: The original model.
        components_to_quantize: List of module names to quantize (e.g., "attn.c_attn", "mlp.c_fc").
    """
    for name, module in model.named_modules():
        if any(comp in name for comp in components_to_quantize):
            if isinstance(module, nn.Linear):
                parent_module, child_name = get_parent_module(model, name)
                setattr(parent_module, child_name, QuantizedLinear(module))
                print(f"Quantized layer: {name}")
    return model

def get_parent_module(model, module_name):
    """
    Gets the parent module and the child's attribute name.
    """
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

# Function to evaluate the model
def evaluate_model(model):
    # Measure memory usage
    memory_usage = get_memory_usage()

    # Compute perplexity
    perplexity = compute_perplexity(model, tokenizer, data, batch_size=8)

    # Measure inference latency
    latency = measure_latency(model, tokenizer, data, batch_size=8)

    return memory_usage, perplexity, latency

# Evaluate the original model
model_fp32 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print("Evaluating original FP32 model...")
memory_fp32, perplexity_fp32, latency_fp32 = evaluate_model(model_fp32)
print(f"FP32 Model - Memory: {memory_fp32:.2f} MB, Perplexity: {perplexity_fp32:.2f}, Latency: {latency_fp32:.4f}s")

# Quantize entire model manually
print("\nQuantizing the entire model to INT8...")
model_int8 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_int8 = quantize_model_manually(model_int8)
memory_int8, perplexity_int8, latency_int8 = evaluate_model(model_int8)
torch.save(model_int8.state_dict(), "Fully_quant.pt")
print(f"INT8 Model - Memory: {memory_int8:.2f} MB, Perplexity: {perplexity_int8:.2f}, Latency: {latency_int8:.4f}s")

# Selectively quantize specific components
components_to_quantize = ["mlp.c_fc", "mlp.c_proj"]  # Example: Quantizing FFN layers
print("\nPerforming selective quantization...")
model_selective = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Reload original model
model_selective = selective_quantization(model_selective, components_to_quantize)

# Evaluate the selectively quantized model
memory_selective, perplexity_selective, latency_selective = evaluate_model(model_selective)
torch.save(model_selective.state_dict(), "Selective_quant.pt")
print(f"Selective Quantization - Memory: {memory_selective:.2f} MB, Perplexity: {perplexity_selective:.2f}, Latency: {latency_selective:.4f}s")

# Summary
print("\n--- Summary ---")
print(f"Original Model: Memory: {memory_fp32:.2f} MB, Perplexity: {perplexity_fp32:.2f}, Latency: {latency_fp32:.4f}s")
print(f"Full Quantized Model: Memory: {memory_int8:.2f} MB, Perplexity: {perplexity_int8:.2f}, Latency: {latency_int8:.4f}s")
print(f"Selective Quantized Model: Memory: {memory_selective:.2f} MB, Perplexity: {perplexity_selective:.2f}, Latency: {latency_selective:.4f}s")




#/////////////////////////////////////////////////////////////////////////////////////////////



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import time
from torch.utils.data import DataLoader
import psutil

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the model name
model_name = 'gpt2'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load the Penn Tree Bank (PTB) dataset
dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
data = dataset['sentence'][:3000] 

# Function to compute perplexity
def compute_perplexity(model, tokenizer, texts, batch_size=8):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    dataloader = DataLoader(texts, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Exclude padding tokens from loss calculation
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Scale loss by the number of tokens
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# Function to measure inference latency
def measure_latency(model, tokenizer, texts, batch_size=8):
    model.eval()
    total_time = 0
    total_sentences = 0
    dataloader = DataLoader(texts, batch_size=batch_size)
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()
            total_time += (end_time - start_time)
            total_sentences += len(batch)
    avg_latency = total_time / total_sentences
    return avg_latency

# Function to measure memory usage
def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
    else:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)  # in MB

# Function to load and evaluate the model
def evaluate_model(model_name, tokenizer, quantization_config=None):
    if quantization_config is None:
        # Load the model in FP32
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        # Load the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map='auto'
        )


    # dummy_input = torch.tensor([[tokenizer.eos_token_id]]).to(model.device)
    # with torch.no_grad():
    #     model(dummy_input)

    # Measure memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(dummy_input)
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
    else:
        memory_usage = get_memory_usage()

    # Compute perplexity
    perplexity = compute_perplexity(model, tokenizer, data, batch_size=8)

    # Measure inference latency
    latency = measure_latency(model, tokenizer, data, batch_size=8)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return memory_usage, perplexity, latency

# Evaluate FP32 model
print("Evaluating FP32 model...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
memory_fp32, perplexity_fp32, latency_fp32 = evaluate_model(model_name, tokenizer)
print(f"Memory usage (MB): {memory_fp32:.2f}")
print(f"Perplexity: {perplexity_fp32:.2f}")
print(f"Inference latency per sentence (s): {latency_fp32:.4f}")

# Evaluate 8-bit quantized model
print("\nEvaluating 8-bit quantized model...")
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_8bit,
    device_map='auto'
)

memory_8bit, perplexity_8bit, latency_8bit = evaluate_model(
    model_name, tokenizer, quantization_config=bnb_config_8bit
)
torch.save(model.state_dict(), f"INT_8.pt")
del model
torch.cuda.empty_cache()
print(f"Memory usage (MB): {memory_8bit:.2f}")
print(f"Perplexity: {perplexity_8bit:.2f}")
print(f"Inference latency per sentence (s): {latency_8bit:.4f}")

# Evaluate 4-bit quantized model (FP4)
print("\nEvaluating 4-bit quantized model (FP4)...")
bnb_config_4bit_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='fp4'
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_4bit_fp4,
    device_map='auto'
)
memory_4bit_fp4, perplexity_4bit_fp4, latency_4bit_fp4 = evaluate_model(
    model_name, tokenizer, quantization_config=bnb_config_4bit_fp4
)
torch.save(model.state_dict(), f"INT_4.pt")
del model
torch.cuda.empty_cache()
print(f"Memory usage (MB): {memory_4bit_fp4:.2f}")
print(f"Perplexity: {perplexity_4bit_fp4:.2f}")
print(f"Inference latency per sentence (s): {latency_4bit_fp4:.4f}")

# Evaluate 4-bit quantized model (NF4)
print("\nEvaluating 4-bit quantized model (NF4)...")
bnb_config_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4'
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_nf4,
    device_map='auto'
)
memory_nf4, perplexity_nf4, latency_nf4 = evaluate_model(
    model_name, tokenizer, quantization_config=bnb_config_nf4
)
torch.save(model.state_dict(), f"NF4.pt")
del model
torch.cuda.empty_cache()
print(f"Memory usage (MB): {memory_nf4:.2f}")
print(f"Perplexity: {perplexity_nf4:.2f}")
print(f"Inference latency per sentence (s): {latency_nf4:.4f}")

# Summary of results
print("\n--- Summary of Quantization Results ---")
print(f"FP32 model - Memory (MB): {memory_fp32:.2f}, Perplexity: {perplexity_fp32:.2f}, Latency (s): {latency_fp32:.4f}")
print(f"8-bit model - Memory (MB): {memory_8bit:.2f}, Perplexity: {perplexity_8bit:.2f}, Latency (s): {latency_8bit:.4f}")
print(f"4-bit FP4 model - Memory (MB): {memory_4bit_fp4:.2f}, Perplexity: {perplexity_4bit_fp4:.2f}, Latency (s): {latency_4bit_fp4:.4f}")
print(f"4-bit NF4 model - Memory (MB): {memory_nf4:.2f}, Perplexity: {perplexity_nf4:.2f}, Latency (s): {latency_nf4:.4f}")
