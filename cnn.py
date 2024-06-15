# CNN using batchnorm

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_hidden = 500
n_layer = 4

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { c:i for i, c in enumerate(chars) }
itos = { i:c for i, c in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]

class CustomBatchNorm1d(nn.Module):
  def __init__(self, nodes):
    super().__init__()
    self.batch_norm = nn.BatchNorm1d(nodes)

  def forward(self, x):
    x = x.transpose(1, 2)
    B, C, T = x.shape
    x = self.batch_norm(x)
    x = x.transpose(2, 1)

    return x

class Block(nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_in, n_out),
        CustomBatchNorm1d(n_out),
        nn.ReLU()
    )
  
  def forward(self, x):
    return self.net(x)
  
# 

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.char_embedding_table = nn.Embedding(vocab_size, n_embd)

    self.first_layer = Block(n_embd, n_hidden)
    self.middle_layer = nn.Sequential(*[Block(n_hidden, n_hidden) for _ in range(n_layer)])
    self.final_layer = Block(n_hidden, vocab_size)
    self.ln = nn.Linear(vocab_size, vocab_size)
    self.apply(self._init_weights)

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    x = self.char_embedding_table(idx) # B, T, C
    
    x = self.first_layer(x)
    x = self.middle_layer(x)
    x = self.final_layer(x)
    logits = self.ln(x)

    if targets == None: #test
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # B, T, C
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      self.eval()
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
  
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)

  return x, y

model = CNN()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))