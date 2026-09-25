import torch
from cs336_basics.transformer import softmax

END = "<|endoftext|>"

def decode(model, tokenizer, prompt: str, max_tokens: int, temperature: float=1.0, top_p=10, device='mps'):
  encoded_prompt = tokenizer.encode(prompt)
  encoded_prompt = torch.tensor(encoded_prompt, device=device)
  encoded_prompt = encoded_prompt.unsqueeze(0)

  generated = 0

  end_token_id = tokenizer.encode(END)

  out = []

  while generated < max_tokens and (len(out) == 0 or out[-1] != end_token_id):
    logits = model(encoded_prompt)

    scaled_logits = logits/temperature

    soft = softmax(scaled_logits, -1)

    sorted_probs, sorted_indices = torch.sort(soft, descending=True, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    
    filtered_probs = softmax(logits, dim=-1)

    pred = torch.multinomial(filtered_probs[:,-1,:], num_samples=1)

    out.append(pred.item())

    encoded_prompt = torch.cat((encoded_prompt, pred), -1)

    print(tokenizer.decode(out))

  print(tokenizer.decode(out))
  return tokenizer.decode(out)


    

