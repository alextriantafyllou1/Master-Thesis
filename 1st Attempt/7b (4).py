#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
get_ipython().system('pip install transformers')


# In[ ]:


import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0))


# In[4]:


import torch
print("Is CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# In[7]:


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


# In[1]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Hugging Face repo for Meditron-7b
model_name = "epfl-llm/meditron-7b"
token = "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"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Example input
input_text = "What is pneumonia?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_length=50)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Response:", response)


# In[12]:


import accelerate
print(accelerate.__version__)


# In[11]:


get_ipython().system('pip install --upgrade accelerate')


# In[2]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Hugging Face repo for Meditron-7b
model_name = "epfl-llm/meditron-70b"
token = "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"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Example input
input_text = "What is pneumonia?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_length=50)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Response:", response)


# In[6]:


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    torch_dtype=torch.float16,
    device_map="auto"  # Spreads the model across multiple GPUs
)


# In[ ]:


import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("GPU is available. Details:")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is NOT available. Using CPU.")


# In[ ]:


import torch

# Create a tensor and move it to GPU
tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
print("Tensor on GPU:", tensor)

# Perform a calculation on the GPU
result = tensor * 2
print("Result of GPU computation:", result)


# In[4]:


def process_query(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
        )
    end_time = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, end_time - start_time


# In[5]:


def load_model(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], token=HF_TOKEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id globally
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_name],
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model


# In[8]:


tokenizer.pad_token_id = tokenizer.eos_token_id


# In[10]:


from transformers import AutoTokenizer

# Define model name and token
model_name = "epfl-llm/meditron-7b"
token = "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"  # Replace with your actual Hugging Face token

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

# Fix the pad_token_id issue
tokenizer.pad_token_id = tokenizer.eos_token_id


# In[6]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from concurrent.futures import ThreadPoolExecutor

# Model Configuration
MODEL_NAME = "Meditron-7B"
MODEL_PATH = "epfl-llm/meditron-7b"
TOKEN = "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"

# Load the model
def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        token=TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Fix: Set pad_token to eos_token if it is not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def process_query(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            pad_token_id=tokenizer.pad_token_id  # Explicitly set pad_token_id
        )
    end_time = time.time()
    duration = end_time - start_time
    return tokenizer.decode(outputs[0], skip_special_tokens=True), duration


# Function to simulate concurrent users
def run_trials(queries, concurrent_users=1):
    tokenizer, model = load_model()
    print(f"Running {MODEL_NAME} with {concurrent_users} concurrent users...")

    def run_single_query(query):
        _, duration = process_query(model, tokenizer, query)
        return duration

    times = []
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = executor.map(run_single_query, queries)
        times.extend(results)

    avg_time = sum(times) / len(times)
    print(f"Average Response Time ({concurrent_users} users): {avg_time:.2f}s")
    return avg_time

# Main function
def main():
    queries = ["What is pneumonia?", "Explain flu symptoms.", "How to treat COVID-19?"] * 10  # Test load
    results = {}

    for load in [1, 2, 4, 8, 16]:
        avg_time = run_trials(queries, concurrent_users=load)
        results[load] = avg_time

    # Save results
    with open("results_meditron_7b.txt", "w") as f:
        f.write(str(results))
    print("Results saved to results_meditron_7b.txt.")

if __name__ == "__main__":
    main()


# In[ ]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from concurrent.futures import ThreadPoolExecutor

# Model Configuration
MODEL_NAME = "Meditron-70B"
MODEL_PATH = "epfl-llm/meditron-70b"
TOKEN = "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"

# Load the model
def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        token=TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Fix: Set pad_token to eos_token if it is not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


# Function to process a single query
def process_query(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            pad_token_id=tokenizer.pad_token_id  # Explicitly set pad_token_id
        )
    end_time = time.time()
    duration = end_time - start_time
    return tokenizer.decode(outputs[0], skip_special_tokens=True), duration


# Function to simulate concurrent users
def run_trials(queries, concurrent_users=1):
    tokenizer, model = load_model()
    print(f"Running {MODEL_NAME} with {concurrent_users} concurrent users...")

    def run_single_query(query):
        _, duration = process_query(model, tokenizer, query)
        return duration

    times = []
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = executor.map(run_single_query, queries)
        times.extend(results)

    avg_time = sum(times) / len(times)
    print(f"Average Response Time ({concurrent_users} users): {avg_time:.2f}s")
    return avg_time


# Main function
def main():
    queries = ["What is pneumonia?", "Explain flu symptoms.", "How to treat COVID-19?"] * 10  # Test load
    results = {}

    for load in [1, 2, 4, 8, 16]:
        avg_time = run_trials(queries, concurrent_users=load)
        results[load] = avg_time

    # Save results
    with open("results_meditron_70b.txt", "w") as f:
        f.write(str(results))
    print("Results saved to results_meditron_70b.txt.")

if __name__ == "__main__":
    main()
##################################### Next step ####################################


# In[2]:


import torch

def check_gpus():
    # Check total number of GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        # Loop through each GPU and print its details
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        
        # Verify if tensors can be sent to all GPUs
        try:
            for i in range(num_gpus):
                device = torch.device(f"cuda:{i}")
                tensor = torch.randn(1).to(device)  # Create a tensor on each GPU
                print(f"Tensor successfully created on GPU {i}: {tensor}")
        except Exception as e:
            print(f"Error when accessing GPU {i}: {e}")
    else:
        print("No GPU detected. Using CPU.")

if __name__ == "__main__":
    check_gpus()


# In[4]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model and Token Configuration
MODEL_PATH = "epfl-llm/meditron-7b"
TOKEN = "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"

# Load the model and tokenizer
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        token=TOKEN,
        torch_dtype=torch.float16,  # Mixed precision
        device_map="auto"           # Automatically allocate to GPU
    )
    return tokenizer, model

# Function to Test Memory for Given Batch Size and Sequence Length
def test_hyperparameters(model, tokenizer, batch_size, seq_length):
    try:
        print(f"Testing Batch Size: {batch_size}, Sequence Length: {seq_length}...")
        input_text = "Test input sentence. " * (seq_length // 5)  # Simulate input text
        inputs = tokenizer(
            [input_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_length
        ).to("cuda")

        # Perform a forward pass with max_new_tokens
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Replace max_length with max_new_tokens
                pad_token_id=tokenizer.pad_token_id
            )

        print(f"Success: Batch Size {batch_size}, Sequence Length {seq_length}")
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"OOM Error: Batch Size {batch_size}, Sequence Length {seq_length}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


# Main Function to Test Combinations
def main():
    tokenizer, model = load_model()

    # Range of Hyperparameters to Test
    batch_sizes = [1, 2, 4, 8, 16]          # Micro batch sizes
    sequence_lengths = [128, 256, 512, 1024, 2048]  # Sequence lengths

    results = []
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            success = test_hyperparameters(model, tokenizer, batch_size, seq_length)
            results.append((batch_size, seq_length, success))
            if not success:
                break  # Stop increasing seq_length if OOM

    # Print Summary of Results
    print("\nTest Results:")
    for batch_size, seq_length, success in results:
        status = "Pass" if success else "Fail (OOM)"
        print(f"Batch Size: {batch_size}, Sequence Length: {seq_length} => {status}")

if __name__ == "__main__":
    main()


# In[9]:


import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Environment variable to suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model Configuration
MODEL_NAME = "Meditron-7B"
MODEL_PATH = "epfl-llm/meditron-7b"
TOKEN = "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"

# Preload tokenizer and model
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    token=TOKEN,
    torch_dtype=torch.float16,
    device_map="auto"
)
# Fix: Set pad_token to eos_token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to log GPU utilization using nvidia-smi
def log_gpu_utilization():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.total,memory.used", "--format=csv,nounits,noheader"],
            stderr=subprocess.DEVNULL
        )
        utilization = output.decode("utf-8").strip().split("\n")[0]
        print(f"GPU Utilization: {utilization}")
    except Exception as e:
        print(f"Failed to retrieve GPU utilization: {e}")

# Function to process a single query
def process_query(query):
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.time()
    duration = end_time - start_time

    # Log GPU Memory Usage
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    log_gpu_utilization()

    return tokenizer.decode(outputs[0], skip_special_tokens=True), duration

# Function to simulate concurrent users
def run_trials(queries, concurrent_users=1):
    print(f"Running {MODEL_NAME} with {concurrent_users} concurrent users...")
    times = []

    def run_single_query(query):
        _, duration = process_query(query)
        return duration

    # Run queries in parallel
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = executor.map(run_single_query, queries)
        times.extend(results)

    avg_time = sum(times) / len(times)
    print(f"Average Response Time ({concurrent_users} users): {avg_time:.2f}s")
    return avg_time

# Main function
def main():
    queries = ["What is pneumonia?", "Explain flu symptoms.", "How to treat COVID-19?"] * 10  # Test load
    results = {}

    for load in [1, 2, 4, 8, 16]:
        avg_time = run_trials(queries, concurrent_users=load)
        results[load] = avg_time

    # Save results
    with open("results_meditron_7b.txt", "w") as f:
        f.write(str(results))
    print("Results saved to results_meditron_7b.txt.")

if __name__ == "__main__":
    main()


# In[ ]:




