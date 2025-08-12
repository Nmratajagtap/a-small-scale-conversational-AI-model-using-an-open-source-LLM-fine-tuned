# a-small-scale-conversational-AI-model-using-an-open-source-LLM-fine-tuned
Build a small-scale conversational AI model using an open-source LLM fine-tuned for one Indian language + English, demonstrating intent detection and context retention. Submit code, README, and sample outputs.
Choose a suitable open-source LLM (e.g., LLaMA 2, Mistral, or BLOOM) and set up the necessary environment in Colab, including installing required libraries and ensuring GPU access.

Reasoning: Choose a suitable open-source LLM and set up the necessary environment in Colab, including installing required libraries and ensuring GPU access. This involves selecting an LLM, installing libraries like transformers and torch, and verifying GPU availability.

# Choose an open-source LLM: Mistral-7B is a good choice for fine-tuning due to its size and performance.
# It is also known to be good with multiple languages.

import torch

gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

if not gpu_available:
    print("Please enable GPU in Colab runtime settings.")
else:
    print("GPU available, proceeding with library installation.")
    !pip install transformers torch accelerate bitsandbytes

    Prepare the dataset
    Prepare the dataset for fine-tuning the LLM.

Reasoning: Create a synthetic dataset for intent detection and context retention in Hindi and English, structuring it for conversational models.

import pandas as pd

# Create a list to hold conversational turns
conversations = []

# Example conversations demonstrating intent detection and context retention
# Conversation 1: Intent - Booking a train ticket, Context - Destination and date
conversations.append({
    "turns": [
        {"role": "user", "text": "मुझे दिल्ली से मुंबई के लिए ट्रेन टिकट बुक करना है।"}, # Hindi: I want to book a train ticket from Delhi to Mumbai.
        {"role": "assistant", "text": "ज़रूर, किस तारीख के लिए टिकट चाहिए?"}, # Hindi: Sure, for which date do you need the ticket?
        {"role": "user", "text": "अगले सोमवार के लिए।"}, # Hindi: For next Monday.
        {"role": "assistant", "text": "ठीक है, अगले सोमवार के लिए दिल्ली से मुंबई की ट्रेनें देख रहा हूँ।"}, # Hindi: Okay, looking for trains from Delhi to Mumbai for next Monday.
    ]
})

# Conversation 2: Intent - Asking about weather, Context - Specific city
conversations.append({
    "turns": [
        {"role": "user", "text": "What is the weather like in London today?"},
        {"role": "assistant", "text": "Let me check the weather for London. It is currently cloudy with a temperature of 15 degrees Celsius."},
        {"role": "user", "text": "And tomorrow?"}, # Context: Still asking about London's weather
        {"role": "assistant", "text": "The forecast for tomorrow in London is sunny with a high of 18 degrees Celsius."},
    ]
})

# Conversation 3: Intent - Ordering food, Context - Adding items to an order
conversations.append({
    "turns": [
        {"role": "user", "text": "I want to order a pizza."},
        {"role": "assistant", "text": "What kind of pizza would you like?"},
        {"role": "user", "text": "A large pepperoni pizza."},
        {"role": "assistant", "text": "Okay, a large pepperoni pizza. Anything else?"},
        {"role": "user", "text": "Yes, add a coke."}, # Context: Adding to the previous order
        {"role": "assistant", "text": "Adding a coke to your order."},
    ]
})

# Conversation 4: Intent - Asking for directions, Context - Starting point
conversations.append({
    "turns": [
        {"role": "user", "text": "मुझे लाल किले तक जाने का रास्ता बताओ।"}, # Hindi: Tell me the way to Red Fort.
        {"role": "assistant", "text": "आप अभी कहाँ हैं?"}, # Hindi: Where are you now?
        {"role": "user", "text": "कनॉट प्लेस में।"}, # Hindi: In Connaught Place.
        {"role": "assistant", "text": "कनॉट प्लेस से लाल किले तक जाने के लिए..."}, # Hindi: To go from Connaught Place to Red Fort...
    ]
})

# Convert to a DataFrame
df_conversations = pd.DataFrame(conversations)

# Display the created dataset
display(df_conversations)

Fine-tune the llm
Subtask:
Fine-tune the selected LLM on the prepared dataset. This will involve defining the training process, including hyperparameters and training loops.
Reasoning: Load the pre-trained Mistral-7B model and tokenizer, prepare the dataset by tokenizing it, define the fine-tuning parameters, and set up the training loop using the transformers library, specifically focusing on the Trainer API for simplicity and efficiency. Since GPU is available now, proceed with the full setup and training process.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# 1. Load the pre-trained Mistral-7B model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer, potentially with quantization if needed (though GPU is available)
# Using bfloat16 for potentially better performance if supported by GPU
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare the dataset for fine-tuning
# The dataset is already in df_conversations from the previous step.
# We need to format it into a single sequence per conversation for causal language modeling.

def format_conversation(conversation):
    formatted_text = ""
    for turn in conversation["turns"]:
        role = turn["role"]
        text = turn["text"]
        # Simple formatting: [ROLE] text
        formatted_text += f"[{role.upper()}] {text} "
    return formatted_text.strip()

df_conversations['formatted_text'] = df_conversations['turns'].apply(format_conversation)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_conversations)

# Tokenize the dataset
def tokenize_function(examples):
    # Add a small amount of padding to avoid issues with different conversation lengths
    # and ensure batching works correctly. Max length should be reasonable for the model.
    return tokenizer(examples["formatted_text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["turns", "formatted_text", "__index_level_0__"])

# Set format for PyTorch
tokenized_dataset.set_format("torch")

# 3. Define the fine-tuning parameters
training_args = TrainingArguments(
    output_dir="./results",  # Output directory
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size per device during training
    gradient_accumulation_steps=4, # Accumulate gradients over 4 steps to simulate a larger batch size
    learning_rate=2e-5,  # Learning rate
    weight_decay=0.01,  # Strength of weight decay
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none", # Disable reporting to external services
)

# 4. Set up the training loop using the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("Starting training...")
trainer.train()
print("Training finished.")

# 5. Save the fine-tuned model and tokenizer
output_model_dir = "./fine-tuned-mistral"
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Fine-tuned model and tokenizer saved to {output_model_dir}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# 1. Load a publicly available pre-trained model and tokenizer
# Switching to google/gemma-2b as it's publicly available and suitable for smaller GPUs
model_name = "google/gemma-2b"

# Check for GPU availability again and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not already set - Gemma uses eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare the dataset for fine-tuning
# The dataset is already in df_conversations.
# We need to format it into a single sequence per conversation for causal language modeling.

def format_conversation(conversation):
    formatted_text = ""
    for turn in conversation["turns"]:
        role = turn["role"]
        text = turn["text"]
        # Simple formatting: [ROLE] text
        formatted_text += f"[{role.upper()}] {text} "
    return formatted_text.strip()

df_conversations['formatted_text'] = df_conversations['turns'].apply(format_conversation)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_conversations)

# Tokenize the dataset
def tokenize_function(examples):
    # Add padding and truncation
    return tokenizer(examples["formatted_text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["turns", "formatted_text", "__index_level_0__"])

# Set format for PyTorch
tokenized_dataset.set_format("torch")

# 3. Define the fine-tuning parameters
training_args = TrainingArguments(
    output_dir="./results",  # Output directory
    num_train_epochs=3,  # Number of training epochs - reduced for small dataset
    per_device_train_batch_size=1,  # Batch size per device during training
    gradient_accumulation_steps=4, # Accumulate gradients over 4 steps
    learning_rate=2e-5,  # Learning rate
    weight_decay=0.01,  # Strength of weight decay
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=1, # Log more frequently for small dataset
    save_steps=100,
    save_total_limit=2,
    report_to="none", # Disable reporting to external services
    push_to_hub=False, # Do not push to Hugging Face Hub
    remove_unused_columns=False # Keep columns for easier debugging if needed
)

# 4. Set up the training loop using the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("Starting training...")
trainer.train()
print("Training finished.")

# 5. Save the fine-tuned model and tokenizer
output_model_dir = "./fine-tuned-gemma"
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Fine-tuned model and tokenizer saved to {output_model_dir}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# 1. Load a publicly available, non-gated pre-trained model and tokenizer
# Switching to gpt2 as it is publicly available and does not require authentication
model_name = "gpt2"

# Check for GPU availability again and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
# Use from_pretrained without torch_dtype=torch.bfloat16 if not on a compatible GPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not already set - GPT2 uses eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare the dataset for fine-tuning
# The dataset is already in df_conversations.
# We need to format it into a single sequence per conversation for causal language modeling.

def format_conversation(conversation):
    formatted_text = ""
    for turn in conversation["turns"]:
        role = turn["role"]
        text = turn["text"]
        # Simple formatting: [ROLE] text
        formatted_text += f"[{role.upper()}] {text} "
    return formatted_text.strip()

df_conversations['formatted_text'] = df_conversations['turns'].apply(format_conversation)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_conversations)

# Tokenize the dataset
def tokenize_function(examples):
    # Add padding and truncation
    return tokenizer(examples["formatted_text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["turns", "formatted_text", "__index_level_0__"])

# Set format for PyTorch
tokenized_dataset.set_format("torch")

# 3. Define the fine-tuning parameters
training_args = TrainingArguments(
    output_dir="./results",  # Output directory
    num_train_epochs=10,  # Increased epochs for very small dataset
    per_device_train_batch_size=1,  # Batch size per device during training
    gradient_accumulation_steps=4, # Accumulate gradients over 4 steps
    learning_rate=5e-5,  # Learning rate
    weight_decay=0.01,  # Strength of weight decay
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=1, # Log more frequently for small dataset
    save_steps=100,
    save_total_limit=2,
    report_to="none", # Disable reporting to external services
    push_to_hub=False, # Do not push to Hugging Face Hub
    remove_unused_columns=False # Keep columns for easier debugging if needed
)

# 4. Set up the training loop using the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("Starting training...")
trainer.train()
print("Training finished.")

# 5. Save the fine-tuned model and tokenizer
output_model_dir = "./fine-tuned-gpt2"
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Fine-tuned model and tokenizer saved to {output_model_dir}")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# 1. Load a publicly available, non-gated pre-trained model and tokenizer
model_name = "gpt2"

# Check for GPU availability again and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not already set - GPT2 uses eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare the dataset for fine-tuning
# Debugging the format_conversation function
def format_conversation(conversation):
    print(f"Input conversation type: {type(conversation)}") # Debug print
    formatted_text = ""
    # Assuming conversation is a list of turn dictionaries based on df_conversations structure
    for turn in conversation: # Iterate directly if conversation is the list of turns
        role = turn["role"]
        text = turn["text"]
        # Simple formatting: [ROLE] text
        formatted_text += f"[{role.upper()}] {text} "
    return formatted_text.strip()

# The dataframe df_conversations already exists from the previous step
# df_conversations['formatted_text'] = df_conversations['turns'].apply(format_conversation) # This line caused the error

# Let's re-apply the formatting using a more robust method or inspect the data
# Based on the traceback, it seems apply is passing the list of turns directly to the function, not the entire row dict.
df_conversations['formatted_text'] = df_conversations['turns'].apply(format_conversation)


# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_conversations)

# Tokenize the dataset
def tokenize_function(examples):
    # Add padding and truncation
    return tokenizer(examples["formatted_text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["turns", "formatted_text", "__index_level_0__"])

# Set format for PyTorch
tokenized_dataset.set_format("torch")

# 3. Define the fine-tuning parameters
training_args = TrainingArguments(
    output_dir="./results",  # Output directory
    num_train_epochs=10,  # Increased epochs for very small dataset
    per_device_train_batch_size=1,  # Batch size per device during training
    gradient_accumulation_steps=4, # Accumulate gradients over 4 steps
    learning_rate=5e-5,  # Learning rate
    weight_decay=0.01,  # Strength of weight decay
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=1, # Log more frequently for small dataset
    save_steps=100,
    save_total_limit=2,
    report_to="none", # Disable reporting to external services
    push_to_hub=False, # Do not push to Hugging Face Hub
    remove_unused_columns=False # Keep columns for easier debugging if needed
)

# 4. Set up the training loop using the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("Starting training...")
trainer.train()
print("Training finished.")

# 5. Save the fine-tuned model and tokenizer
output_model_dir = "./fine-tuned-gpt2"
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Fine-tuned model and tokenizer saved to {output_model_dir}")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# 1. Load a publicly available, non-gated pre-trained model and tokenizer
model_name = "gpt2"

# Check for GPU availability again and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not already set - GPT2 uses eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare the dataset for fine-tuning
# The dataset is already in df_conversations.
# We need to format it into a single sequence per conversation for causal language modeling.

def format_conversation(conversation):
    formatted_text = ""
    # Iterate directly if conversation is the list of turns
    for turn in conversation:
        role = turn["role"]
        text = turn["text"]
        # Simple formatting: [ROLE] text
        formatted_text += f"[{role.upper()}] {text} "
    return formatted_text.strip()

# The dataframe df_conversations already exists from the previous step
df_conversations['formatted_text'] = df_conversations['turns'].apply(format_conversation)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_conversations)

# Tokenize the dataset
def tokenize_function(examples):
    # Add padding and truncation
    return tokenizer(examples["formatted_text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["turns", "formatted_text"])

# Add 'labels' column which is a copy of 'input_ids' for causal language modeling
tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)


# Set format for PyTorch
tokenized_dataset.set_format("torch")

# 3. Define the fine-tuning parameters
training_args = TrainingArguments(
    output_dir="./results",  # Output directory
    num_train_epochs=10,  # Increased epochs for very small dataset
    per_device_train_batch_size=1,  # Batch size per device during training
    gradient_accumulation_steps=4, # Accumulate gradients over 4 steps
    learning_rate=5e-5,  # Learning rate
    weight_decay=0.01,  # Strength of weight decay
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=1, # Log more frequently for small dataset
    save_steps=100,
    save_total_limit=2,
    report_to="none", # Disable reporting to external services
    push_to_hub=False, # Do not push to Hugging Face Hub
    remove_unused_columns=False # Keep columns for easier debugging if needed
)

# 4. Set up the training loop using the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("Starting training...")
trainer.train()
print("Training finished.")

# 5. Save the fine-tuned model and tokenizer
output_model_dir = "./fine-tuned-gpt2"
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Fine-tuned model and tokenizer saved to {output_model_dir}")

Subtask:
Develop code to integrate intent detection and context retention mechanisms into the fine-tuned model.

Reasoning: Load the fine-tuned model and tokenizer and define a function to generate responses based on conversation history and new input.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the fine-tuned model and tokenizer
output_model_dir = "./fine-tuned-gpt2"
try:
    model = AutoModelForCausalLM.from_pretrained(output_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded model and tokenizer from {output_model_dir}")
    print(f"Model moved to device: {device}")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    model = None
    tokenizer = None
    device = None

# Set padding token if not already set - GPT2 uses eos_token as pad_token
if tokenizer and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token for tokenizer.")


# 2. Create a function for generating responses
def generate_response(conversation_history, new_user_input, model, tokenizer, device, max_length=200, num_return_sequences=1, no_repeat_ngram_size=3, temperature=0.7):
    """
    Generates a response from the model given a conversation history and new user input.

    Args:
        conversation_history (list): A list of dictionaries, where each dictionary
                                     represents a turn with 'role' and 'text' keys.
        new_user_input (str): The new input from the user.
        model: The loaded fine-tuned model.
        tokenizer: The loaded tokenizer.
        device: The device (cuda or cpu) to run the model on.
        max_length (int): The maximum length of the generated sequence.
        num_return_sequences (int): The number of sequences to generate.
        no_repeat_ngram_size (int): The size of n-grams that should not be repeated.
        temperature (float): Controls the randomness in generation.

    Returns:
        str: The generated response string.
    """
    if model is None or tokenizer is None or device is None:
        return "Error: Model or tokenizer not loaded."

    # 3. Format the conversation history and new user input
    formatted_input = ""
    for turn in conversation_history:
        formatted_input += f"[{turn['role'].upper()}] {turn['text']} "
    formatted_input += f"[USER] {new_user_input}"

    print(f"Formatted input for generation: {formatted_input}")

    # 4. Tokenize the formatted input string
    # Add return_tensors='pt' to get PyTorch tensors
    # Add padding='longest' for batching if needed (though here it's a single input)
    # Add truncation=True to handle inputs longer than max_length
    inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

    print(f"Input token IDs: {inputs['input_ids']}")
    print(f"Attention mask: {inputs['attention_mask']}")


    # 5. Use the loaded model to generate a response
    # Set pad_token_id for generation
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"Using pad_token_id: {pad_token_id}")

    try:
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            pad_token_id=pad_token_id # Ensure pad_token_id is set
        )
        print("Generation successful.")
    except Exception as e:
        print(f"Error during generation: {e}")
        return "Error during response generation."


    # 6. Extract the generated text
    # Decode the generated tokens, skipping the input tokens
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(f"Raw generated text: {generated_text}")


    # 7. Post-process the generated text
    # Simple post-processing: Find the first "[ASSISTANT]" tag and take everything after it.
    # If no "[ASSISTANT]" tag, return a default response or the raw text.
    assistant_tag = "[ASSISTANT]"
    if assistant_tag in generated_text:
        # Find the first occurrence of the assistant tag after the original input
        # This is a heuristic and might need refinement based on model output patterns
        input_length = len(formatted_input)
        # Decode the input part to find its length in the generated text
        decoded_input = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        input_end_index = generated_text.find(decoded_input) + len(decoded_input)

        # Search for the assistant tag after the input
        assistant_tag_index = generated_text.find(assistant_tag, input_end_index)

        if assistant_tag_index != -1:
             # Extract text after the assistant tag and strip leading/trailing whitespace
            response = generated_text[assistant_tag_index + len(assistant_tag):].strip()
        else:
             # If tag not found after input, maybe it's at the beginning, take everything after first tag
             assistant_tag_index = generated_text.find(assistant_tag)
             if assistant_tag_index != -1:
                  response = generated_text[assistant_tag_index + len(assistant_tag):].strip()
             else:
                  response = "Could not find assistant response tag." # Fallback if tag isn't found at all
    else:
        response = "Could not find assistant response tag in generated text." # Fallback if tag isn't found at all


    # Further cleaning: Remove potential trailing incomplete sentences or repeated tags
    # This is a basic example, more sophisticated cleaning might be needed
    if response.endswith("[USER]"):
        response = response[:-len("[USER]")].strip()
    # Add more cleaning rules as needed based on observed model outputs

    # 8. Return the generated response string
    return response

# 1. Create a small test set (distinct from training data)
test_conversations = [
    {
        "description": "Test Case 1: Simple question (English)",
        "turns": [
            {"role": "user", "text": "What is the capital of France?"}
        ]
    },
    {
        "description": "Test Case 2: Simple question (Hindi)",
        "turns": [
            {"role": "user", "text": "भारत की राजधानी क्या है?"} # Hindi: What is the capital of India?
        ]
    },
    {
        "description": "Test Case 3: Context Retention (Previous turn)",
        "turns": [
            {"role": "user", "text": "I want to know about the weather in New York."},
            {"role": "assistant", "text": "Let me check the weather for New York."},
            {"role": "user", "text": "Is it sunny today?"} # Context: still about New York weather
        ]
    },
     {
        "description": "Test Case 4: Context Retention (Multiple turns)",
        "turns": [
            {"role": "user", "text": "मुझे दिल्ली से चेन्नई के लिए फ्लाइट चाहिए।"}, # Hindi: I need a flight from Delhi to Chennai.
            {"role": "assistant", "text": "किस तारीख को?"}, # Hindi: On which date?
            {"role": "user", "text": "अगले महीने की 15 तारीख को।"} # Hindi: On the 15th of next month.
        ]
    },
     {
        "description": "Test Case 5: Intent Detection (Food Ordering)",
        "turns": [
            {"role": "user", "text": "Can I order a burger?"}
        ]
    }
]

# Store evaluation results
evaluation_results = []

# 2. Iterate through the test set and evaluate the model's responses
print("\n--- Evaluating the fine-tuned model on test set ---")

if model and tokenizer:
    for test_case in test_conversations:
        print(f"\n--- {test_case['description']} ---")
        conversation_history = []
        case_eval = {"description": test_case["description"], "turns_evaluation": []}

        for i, turn in enumerate(test_case["turns"]):
            if turn["role"] == "user":
                new_user_input = turn["text"]
                print(f"User: {new_user_input}")

                # Generate response
                response = generate_response(conversation_history, new_user_input, model, tokenizer, device)
                print(f"Assistant (Generated): {response}")

                # Store for evaluation
                case_eval["turns_evaluation"].append({
                    "user_input": new_user_input,
                    "generated_response": response,
                    "expected_intent": "N/A", # Manually determine expected intent
                    "expected_context_use": "N/A", # Manually determine expected context use
                    "fluency_score": "N/A", # Manual evaluation
                    "intent_score": "N/A", # Manual evaluation
                    "context_score": "N/A", # Manual evaluation
                    "extractable_response": "N/A" # Check if post-processing extracted a response
                })

            # Update conversation history with the current turn
            conversation_history.append(turn)

        evaluation_results.append(case_eval)

else:
    print("\nSkipping evaluation as model or tokenizer failed to load.")

# Manually evaluate the results based on the print output and fill in the evaluation_results structure.
# This manual evaluation step cannot be automated by the code interpreter.
# After running the code, you would review the 'Assistant (Generated)' outputs and fill in the scores.

# Example of how to manually add scores after reviewing the output:
# evaluation_results[0]["turns_evaluation"][0]["fluency_score"] = "Poor"
# evaluation_results[0]["turns_evaluation"][0]["intent_score"] = "Poor"
# evaluation_results[0]["turns_evaluation"][0]["context_score"] = "N/A" # N/A for first turn without history
# evaluation_results[0]["turns_evaluation"][0]["extractable_response"] = "No" # Based on previous observation

# Print a placeholder for manual evaluation summary instruction
print("\n--- Manual Evaluation Required ---")
print("Please review the generated responses above for each test case.")
print("Manually assess Conversational Fluency, Intent Detection, and Context Retention.")
print("Also, note if the response extracted by the post-processing logic is meaningful (check for 'Could not find assistant response tag').")
print("Record your observations and scores (e.g., Good, Partial, Poor) for each turn in the 'evaluation_results' structure.")
print("Then, summarize the overall evaluation results.")

# Note: The following code block will only print the structure, it will not contain the actual manual scores.
# The user (or the next automated step) is expected to manually fill in the scores based on the output.

## Demonstrate the model
# 1. Define example conversational inputs distinct from training/evaluation sets.
# These examples are crafted to test intent and context, acknowledging the model's likely limitations.

sample_conversations = [
    {
        "description": "Sample 1: Basic Hindi Question & Follow-up (Testing basic response and context)",
        "turns": [
            {"role": "user", "text": "आज मौसम कैसा है?"}, # Hindi: How is the weather today?
            {"role": "assistant", "text": ""}, # Placeholder for model response
            {"role": "user", "text": "क्या कल बारिश होगी?"}, # Hindi: Will it rain tomorrow? (Context: weather inquiry)
            {"role": "assistant", "text": ""}, # Placeholder for model response
        ]
    },
    {
        "description": "Sample 2: English Food Order & Modification (Testing intent and context retention)",
        "turns": [
            {"role": "user", "text": "I'd like to order a coffee."},
            {"role": "assistant", "text": ""}, # Placeholder for model response
            {"role": "user", "text": "Make it a large one."}, # Context: modifying the coffee order
            {"role": "assistant", "text": ""}, # Placeholder for model response
        ]
    },
    {
        "description": "Sample 3: Mixed Language Directions Query (Testing handling both languages and context)",
        "turns": [
            {"role": "user", "text": "मुझे इंडिया गेट जाना है।"}, # Hindi: I want to go to India Gate.
            {"role": "assistant", "text": ""}, # Placeholder for model response
            {"role": "user", "text": "from my current location?"}, # English: from my current location? (Context: directions to India Gate)
            {"role": "assistant", "text": ""}, # Placeholder for model response
        ]
    },
     {
        "description": "Sample 4: Simple English Greeting & Query",
        "turns": [
            {"role": "user", "text": "Hi there!"},
            {"role": "assistant", "text": ""}, # Placeholder for model response
            {"role": "user", "text": "Tell me a joke."}, # New intent
            {"role": "assistant", "text": ""}, # Placeholder for model response
        ]
    }
]

# 2. Simulate the turns by calling the generate_response function sequentially.
# 3. Print the user input and the model's generated response for each turn.
# 4. Add brief comments or descriptions.

if model and tokenizer:
    print("\n--- Generating Sample Outputs ---")

    for sample_case in sample_conversations:
        print(f"\n--- {sample_case['description']} ---")
        conversation_history = []

        for i, turn in enumerate(sample_case["turns"]):
            if turn["role"] == "user":
                new_user_input = turn["text"]
                print(f"User: {new_user_input}")

                # Generate response
                response = generate_response(conversation_history, new_user_input, model, tokenizer, device)

                # Update the assistant turn in the sample_case with the generated response
                # Note: This modifies the original list, useful for building history
                if i + 1 < len(sample_case["turns"]) and sample_case["turns"][i+1]["role"] == "assistant":
                     sample_case["turns"][i+1]["text"] = response
                     print(f"Assistant: {response}")
                else:
                     # Handle cases where there isn't an expected assistant turn immediately after
                     print(f"Assistant (Generated): {response}")


                # Update conversation history with the current user turn and the generated assistant response
                conversation_history.append({"role": "user", "text": new_user_input})
                # Only add the generated response to history if it's meaningful,
                # but given previous results, we'll add it regardless to see how it affects future turns.
                conversation_history.append({"role": "assistant", "text": response})

            elif turn["role"] == "assistant" and turn["text"] != "":
                 # This branch is mainly to print the assistant's response that was just filled in
                 # The actual generation happens when the role is "user"
                 pass # Handled in the "user" block above

else:
    print("\nSkipping sample output generation as model or tokenizer failed to load.")
Implement intent detection and context retention


    # Authenticate to Hugging Face (optional, only if needed for private models or datasets)
    # from huggingface_hub import notebook_login
    # notebook_login()

# 9. Test the function with example conversational inputs

if model and tokenizer:
    print("\n--- Testing the response generation function ---")

    # Test Case 1: Simple greeting (Intent detection)
    print("\nTest Case 1: Simple greeting")
    history1 = []
    new_input1 = "Hello!"
    response1 = generate_response(history1, new_input1, model, tokenizer, device)
    print(f"User: {new_input1}")
    print(f"Assistant: {response1}")

    # Test Case 2: Train ticket booking (Intent detection + Context retention - Destination/Origin)
    print("\nTest Case 2: Train ticket booking (Intent + Context)")
    history2 = [
        {"role": "user", "text": "मुझे दिल्ली से मुंबई के लिए ट्रेन टिकट बुक करना है।"}
    ]
    new_input2 = "किस तारीख के लिए?" # Hindi: For which date?
    response2 = generate_response(history2, new_input2, model, tokenizer, device)
    print(f"Conversation History: {history2}")
    print(f"User: {new_input2}")
    print(f"Assistant: {response2}")

    # Test Case 3: Train ticket booking (Context retention - Date)
    print("\nTest Case 3: Train ticket booking (Context - Date)")
    history3 = [
        {"role": "user", "text": "मुझे दिल्ली से मुंबई के लिए ट्रेन टिकट बुक करना है।"},
        {"role": "assistant", "text": "ज़रूर, किस तारीख के लिए टिकट चाहिए?"}
    ]
    new_input3 = "अगले सोमवार के लिए।" # Hindi: For next Monday.
    response3 = generate_response(history3, new_input3, model, tokenizer, device)
    print(f"Conversation History: {history3}")
    print(f"User: {new_input3}")
    print(f"Assistant: {response3}")

    # Test Case 4: Weather query (Intent detection)
    print("\nTest Case 4: Weather query (Intent)")
    history4 = []
    new_input4 = "What is the weather like in Paris today?"
    response4 = generate_response(history4, new_input4, model, tokenizer, device)
    print(f"User: {new_input4}")
    print(f"Assistant: {response4}")

    # Test Case 5: Weather query (Context retention - City)
    print("\nTest Case 5: Weather query (Context - City)")
    history5 = [
        {"role": "user", "text": "What is the weather like in London today?"},
        {"role": "assistant", "text": "Let me check the weather for London. It is currently cloudy with a temperature of 15 degrees Celsius."}
    ]
    new_input5 = "And tomorrow?" # Context: Still asking about London's weather
    response5 = generate_response(history5, new_input5, model, tokenizer, device)
    print(f"Conversation History: {history5}")
    print(f"User: {new_input5}")
    print(f"Assistant: {response5}")

    # Test Case 6: Food ordering (Intent detection + Context retention - Item)
    print("\nTest Case 6: Food ordering (Intent + Context)")
    history6 = [
        {"role": "user", "text": "I want to order a pizza."},
        {"role": "assistant", "text": "What kind of pizza would you like?"}
    ]
    new_input6 = "A large pepperoni pizza."
    response6 = generate_response(history6, new_input6, model, tokenizer, device)
    print(f"Conversation History: {history6}")
    print(f"User: {new_input6}")
    print(f"Assistant: {response6}")

    # Test Case 7: Food ordering (Context retention - Adding item)
    print("\nTest Case 7: Food ordering (Context - Adding item)")
    history7 = [
        {"role": "user", "text": "I want to order a pizza."},
        {"role": "assistant", "text": "What kind of pizza would you like?"},
        {"role": "user", "text": "A large pepperoni pizza."},
        {"role": "assistant", "text": "Okay, a large pepperoni pizza. Anything else?"}
    ]
    new_input7 = "Yes, add a coke." # Context: Adding to the previous order
    response7 = generate_response(history7, new_input7, model, tokenizer, device)
    print(f"Conversation History: {history7}")
    print(f"User: {new_input7}")
    print(f"Assistant: {response7}")

    # Test Case 8: Directions (Intent detection + Context retention - Destination)
    print("\nTest Case 8: Directions (Intent + Context)")
    history8 = [
        {"role": "user", "text": "मुझे लाल किले तक जाने का रास्ता बताओ।"} # Hindi: Tell me the way to Red Fort.
    ]
    new_input8 = "आप अभी कहाँ हैं?" # Hindi: Where are you now?
    response8 = generate_response(history8, new_input8, model, tokenizer, device)
    print(f"Conversation History: {history8}")
    print(f"User: {new_input8}")
    print(f"Assistant: {response8}")

    # Test Case 9: Directions (Context retention - Starting point)
    print("\nTest Case 9: Directions (Context - Starting point)")
    history9 = [
        {"role": "user", "text": "मुझे लाल किले तक जाने का रास्ता बताओ।"}, # Hindi: Tell me the way to Red Fort.
        {"role": "assistant", "text": "आप अभी कहाँ हैं?"} # Hindi: Where are you now?
    ]
    new_input9 = "कनॉट प्लेस में।" # Hindi: In Connaught Place.
    response9 = generate_response(history9, new_input9, model, tokenizer, device)
    print(f"Conversation History: {history9}")
    print(f"User: {new_input9}")
    print(f"Assistant: {response9}")

else:
    print("\nSkipping testing as model or tokenizer failed to load.")
