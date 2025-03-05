from transformers import AutoModelForCausalLM, AutoTokenizer
print('point 1')
model = AutoModelForCausalLM.from_pretrained("upstage/solar-pro-preview-instruct",trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("upstage/solar-pro-preview-instruct")
print('point 2')
prompt = "This is an example script ."
inputs = tokenizer(prompt, return_tensors="pt")
print('point 3')
# Generate
messages = [
    {"role": "user", "content": "Please, introduce yourself."},
]
prompt = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
# Generate text

generate_ids = model.generate(prompt, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
print('point 4')