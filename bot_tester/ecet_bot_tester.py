import torch, os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Load the trained model for testing
model = GPT2LMHeadModel.from_pretrained("../models/fine-tuned_model")
#model = GPT2LMHeadModel.from_pretrained("./gpt2-training/checkpoint-7000")
#model = GPT2LMHeadModel.from_pretrained("./gpt2-trained-model")

# Set the model to evaluation mode
model.eval()

while(1):
    # Get user input
    user_input = input("Enter your message: ")
    
    # Tokenize the user input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    
    # Generate the attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Generate response
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    start_index = response.find("Bot:") + len("Bot:")
    end_index = response.find("User:")

    desired_output = response[start_index:end_index].strip()

    #os.system('cls')

    print("Bot response:-----------------------\n\n", desired_output)
    print("\n--------------------\n")
