import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers.optimization import AdamW

for i in range(3): # after 3 , overfitting might happen
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #model = GPT2LMHeadModel.from_pretrained("./fine-tuned_model") # comment this on first run - Gopi
    #model = GPT2LMHeadModel.from_pretrained("gpt2") # uncomment on first run - Gopi 
    #model = GPT2LMHeadModel.from_pretrained("./gpt2-trained-model")
    #optimizer = AdamW(model.parameters(), lr=1e-6)
    
    # Load and preprocess the training data
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="../dataset/qaDataset.txt",  # Path to training dataset
        block_size=256  
    )
    
    # Load and preprocess the evaluation data
    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="../dataset/eval.txt",  # Path to your evaluation dataset
        block_size=256
    )
    
    # Create the data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="../models/gpt2-training",  # Directory to save the trained model
        overwrite_output_dir=True,
        #evaluation_strategy="epoch",
        num_train_epochs=200,  # Set the desired number of training epochs
        per_device_train_batch_size=1024,  # Set the batch size for training
        per_device_eval_batch_size=64,  # Set the batch size for evaluation
        logging_dir="./logs",  # Directory for storing logs
        weight_decay=0.01,
        #logging_steps=256,  # Log training loss after every specified number of steps
        save_total_limit=2,  # Save the last 2 models checkpoints
        #evaluation_strategy="steps",
        eval_steps=512,  # Evaluate the model after every specified number of steps
        learning_rate=1e-5
    )
    
    # Create the Trainer instance
    '''trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None)
    )'''


   
    if i==0:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    elif i==1:
        model = GPT2LMHeadModel.from_pretrained("../models/gpt2-trained-model")
    else:
        model = GPT2LMHeadModel.from_pretrained("../models/fine-tuned_model")
    #model = GPT2LMHeadModel.from_pretrained("./fine-tuned_model")


    #model = GPT2LMHeadModel.from_pretrained("./fine-tuned_model") # comment this on first run - Gopi
    #model = GPT2LMHeadModel.from_pretrained("gpt2") # uncomment on first run - Gopi 
    #model = GPT2LMHeadModel.from_pretrained("./gpt2-trained-model")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = AdamW(model.parameters(), lr=1e-4)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None)
    )
    # Train the model
    trainer.train()
    
    if i==0:
        # Save the trained model
        model.save_pretrained("../models/gpt2-trained-model") # uncomment on first run and comment on second run hehe - Gopi
    else:
        #save the fine tuned model
        trainer.save_model("../models/fine-tuned_model") # comment this on first run and uncomment from sencond run.
    #trainer.save_model("../models/fine-tuned_model")
    
    print("------------- Training Done ! ---------------- Ite:: " + str(i + 1))
print("\n\nDone ...!\n")