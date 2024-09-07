import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
max_iters = 200
n_embd = 10
dropout = 0.0
block_size = 12
n_head = 2
n_layer = 2
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-3
eval_iters = 50
eval_interval = 20

# Data loading and preparation
with open("all_tswift_lyrics.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[n] for n in l])

data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]


def get_batch(split):

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in ix]
    )  # target is input shifted to right by one position
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def get_losses():
    out = {}
    model.eval()
    for split in ["train", "val"]:  # get the mean of both train and eval loss
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class attention_head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)  # just in case is necessary

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(
            tril == 0, float("-inf")
        )  # wei can be interpreted as logits before applying softmax, if I want to mask out the future, I can do so by filling the future with -inf
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [attention_head(head_size) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


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


class attention_block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.multi_head = MultiHeadAttention(n_head, self.head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head(
            self.ln1(x)
        )  # the '+' for residual connections, layer norm is applied before the multi head attention
        x = x + self.ffwd(
            self.ln2(x)
        )  # the '+' for residual connections, layer norm is applied before the feed forward
        return x


class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[attention_block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(
            n_embd, vocab_size
        )  # project the output of the final block to logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.positional_embedding_table(
            torch.arange(T)
        )  # (T, n_embd), it's like passing to pos_emb [0,1,...T]
        x = tok_emb + pos_emb  # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = (
                logits.shape
            )  # C is the vocab size aka number of classes for cross entropy loss
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)  # targets is (B, T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        """input idx is (B, T) array of indices in the current context"""
        for _ in range(max_tokens):
            # crop idx in order to fit in the model context length (block_size)
            idx_cropped = idx[:, -block_size:]
            # get logits and loss from the model
            logits, loss = self(idx_cropped)
            # focus only on the last logits aka the prediction for the next token
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities over possible next tokens
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def train(model):

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # training loop
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = get_losses()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample data
        xb, yb = get_batch("train")

        # evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    model = GPT()
    model = model.to(device)
    train(model)
