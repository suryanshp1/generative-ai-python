from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("af1tang/personaGPT")

dialog_hx = []

if torch.cuda.is_available():
    model = model.cuda()
## utility functions ##
flatten = lambda l: [item for sublist in l for item in sublist]

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(f">> User: {msg}")
        else:
            print(f"Bot: {msg}")
            print()

def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_length=1000, pad_token=tokenizer.eos_token_id):
    full_msg = model.generate(bot_input_ids, do_sample=True,
                                              top_k=top_k, top_p=top_p, 
                                              max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]

while True:
    userText = input("User: ")  # Prompt is blank so will generate a new line.
    response = f"{tokenizer.eos_token}My name is Suraj, I am 23 years old and I am passionate about movies."
    personas = [response]
    response = f"{tokenizer.eos_token}I am a software engineer."
    personas.append(response)
    response = f"{tokenizer.eos_token}I am single and follow vedanta."
    personas.append(response)
    response = f"{tokenizer.eos_token}I hate talking or interacting with people"
    personas.append(response)
    response = f"{tokenizer.eos_token}The only thing I like is movies."
    personas.append(response)
    response = f"{tokenizer.eos_token}I drink water all day and do nothing."
    personas.append(response)

    personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))


    # encode the user input
    user_inp = tokenizer.encode(tokenizer.eos_token + userText)
    # append to the chat history
    dialog_hx.append(user_inp)

    # generated a response while limiting the total chat history to 1000 tokens, 
    bot_input_ids = to_var([personas + flatten(dialog_hx)]).long()
    msg = generate_next(bot_input_ids)
    dialog_hx.append(msg)
    print(f"Persona: {tokenizer.decode(msg, skip_special_tokens=True)}")