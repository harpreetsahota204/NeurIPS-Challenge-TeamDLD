import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login, HfApi
# import logging
# from pathlib import Path


#logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logger = logging.getLogger(__name__)


def format_row_as_instruction_prompt(example):
    # Check if 'input' key exists and has content
    has_input = example.get('input', None) is not None

    # Define the prompts based on the presence of input
    if has_input:
        primer_prompt = ("Below is an instruction that describes a task, paired with an input "
                         "that provides further context. Write a response that appropriately completes the request.")
        input_template = f"### Input: \n{example['input']}\n\n"
    else:
        primer_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.")
        input_template = ""

    instruction_template = f"### Instruction: \n{example['instruction']}\n\n"

    # Check if 'output' key exists
    if example.get('output', None):
        response_template = f"### Response: \n{example['output']}\n\n"
    else:
        response_template = ""

    return f"{primer_prompt}\n\n{instruction_template}{input_template}{response_template}"

def upload_model_to_hf():
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    api = HfApi() 

    api.upload_folder( 
        folder_path='./decilm6b_open_instruct', 
        repo_id=os.environ["HUGGINGFACE_REPO"],
        repo_type='model', 
    )


def train():
  # Data preparation
  open_instruct_dataset = load_dataset("TeamDLD/neurips_challenge_dataset")
  # filter dataset to rows where the entire context length is less than or equal to 4096,
  # which is the size of the DeciLM-6B context window
  # dataset = open_instruct_dataset.filter(lambda example: (len(example["input"]) + len(example["output"]) + len(example["instruction"])) <= 4096)

  # Only select subst of the datasets
#   total_data_points = len(dataset)
#   sample_size = 5_000
#   random_indices = random.sample(range(total_data_points), sample_size)
  # subset = open_instruct_dataset.select(random_indices)
  subset = open_instruct_dataset

  #logger.info("Prepared subset of training data of size: {0}".format(len(subset)))

  # Model training configs
  model_id = "Deci/DeciLM-6b"

  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )

  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      quantization_config=quantization_config,
      use_cache=False,
      device_map="auto",
      trust_remote_code=True
  )

  model.config.pretraining_tp = 1

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias="none",
      task_type="CAUSAL_LM"
  )

  #logger.info("Preparing model for kbit training")
  model = prepare_model_for_kbit_training(model)
  #logger.info("Prepared model for kbit training LoRA configs")

  args = TrainingArguments(
      output_dir="decilm6b_open_instruct",
      # just for demo purposes
      num_train_epochs=1,
      # trying to max out resources on colab
      per_device_train_batch_size=4,
      gradient_accumulation_steps=10,
      gradient_checkpointing=True,
      optim="paged_adamw_32bit",
      logging_steps=25,
      save_strategy="steps",
      save_steps=100,
      learning_rate=3e-5,
      bf16=True,
      tf32=True,
      max_grad_norm=0.3,
      warmup_ratio=0.03,
      lr_scheduler_type="linear",
      disable_tqdm=False
  )

  model = get_peft_model(model, peft_config)

  max_seq_len = 4096
  trainer = SFTTrainer(
      model=model,
      train_dataset=subset,
      peft_config=peft_config,
      max_seq_length=max_seq_len,
      tokenizer=tokenizer,
      packing=True,
      #formatting_func=format_row_as_instruction_prompt,
      args=args,
  )
  
  #logger.info("Start training")
  print("Start training")
  trainer.train()
  #logger.info("Finished training")
  print("Finished training")


  trainer.save_model()
  # logger.info("Saved model")
  print("Saved model within container instance")

  #logger.info("Uploading model to HuggingFace")
  print("Uploading model to HuggingFace")
  upload_model_to_hf()
  print("Finished uploading model to HuggingFace")




if __name__ == "__main__":
    train()