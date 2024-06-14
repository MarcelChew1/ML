import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' #if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

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
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  idx = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in idx])
  y = torch.stack([data[i+1:i+block_size+1] for i in idx])
  x, y = x.to(device), y.to(device)
  return x, y

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

class EncapsulatedMultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.attn_attr = nn.Linear(n_embd, n_embd * 3)
    self.proj = nn.Linear(n_embd, n_embd)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.attn_attr(x).split(n_embd, dim=2) # keeps dimensions B, T from input, final dim (2) becomes 3 * n_embd composed of q,k,v which are n_heads * head_size in dimension
    
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # initially B, T, C -> B, T, heads, head_size transpose so the num heads is a batch dimension
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

    attn = q @ k.transpose(-2, -1)*k.shape[-1]**-0.5 # B, nh, T, hs @ B, nh, hs, T => B, nh, T, T
    attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # adds batch dim 1, 1 and projects
    attn = F.softmax(attn, dim=-1)
    attn = self.dropout(attn)
    y = attn @ v # B, nh, T, hs
    
    y = y.transpose(1, 2).contiguous().view(B, T, C) # reorder back to original structure heads move to dim = 2
    y = self.dropout(self.proj(y)) # output projection layer

    return y
class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj = nn.Linear(head_size * n_head, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False) # B, T, head_size
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1)*k.shape[-1]**-0.5 # B, T, head_size * B, head_size, T = B, T, T
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out
  
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )
    
  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    # self.sa = MultiHeadAttention(n_head, head_size)
    self.sa = EncapsulatedMultiHeadAttention()
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class GPTLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # make embedding table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.block = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    
    self.apply(self._init_weights)

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_embd = self.token_embedding_table(idx)
    pos_embd = self.position_embedding_table(torch.arange(T, device=device))

    x = tok_embd + pos_embd
    x = self.block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)
    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
  

def train_model():
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  for iter in range(max_iters):

      # every once in a while evaluate the loss on train and val sets
      if iter % eval_interval == 0 or iter == max_iters - 1:
          losses = estimate_loss()
          print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

      # sample a batch of data
      xb, yb = get_batch('train')

      # evaluate the loss
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

model = GPTLanguageModel()
m = model.to(device)

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

train_model()
save_model(m)

def load_model(path='model.pth'):
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model

# m = load_model()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
