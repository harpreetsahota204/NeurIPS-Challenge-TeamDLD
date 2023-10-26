from transformers import AutoTokenizer, AutoModelForCausalLM

# download the model
MODEL = "TeamDLD/decilm6b_open_instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# save the model
save_dir = "models/decilm6b_open_instruct"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print('Download model successful!!')