import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# %matplotlib inline

words = open('names.txt', 'r').read().splitlines()
print(len(words))

chars = ['.'] + sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(f'Vocabulary size: {vocab_size}')

block_size = 3

def build_dataset(words):
    X,Y = [],[]
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtrn, Ytrn = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtst, Ytst = build_dataset(words[n2:])

n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd),           generator=g)
W1 = torch.randn((n_embd*block_size, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden,                      generator=g) * 0.01 # multiply by 0 so that initial loss is not high. closer, but not equal to uniform probability distribution
W2 = torch.randn((n_hidden, vocab_size),        generator=g) * 0.01 # additionally, this helps the issue of "dead neurons" on initialization caused by tanh saturation
b2 = torch.randn(vocab_size,                    generator=g) * 0.0  # - derivative of tanh, 1-x**2, which means if value is extreme, difference will be 0 and product with gradient will be irrelevant

parameters = [C,W1,b1,W2,b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi =[]

for i in range(max_steps):
    # mini-batch construct
    ix = torch.randint(0, Xtrn.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtrn[ix], Ytrn[ix] # batch X,Y

    # forward-pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    hpreact = embcat @ W1 + b1 # hidden layer pre-calcluate
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    # backward-pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    if i % 10000 == 0: # print every 10K steps
        print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')
    lossi.append(loss.log10().item())

@torch.no_grad() # this decorate disables gradient tracking for optimization (can't call .backward())
def split_loss(split):
    x,y = {
        'train': (Xtrn, Ytrn),
        'val': (Xdev, Ydev),
        'test': (Xtst, Ytst)
    }[split]
    emb = C[x] # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
    h = torch.tanh(embcat @ W1 + b1) # (N, n_hidden)
    logits = h @ W2 + b2 # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')


def sample_model():
    g = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(20):
        out = []
        context = [0] * block_size # initialize with all 
        while True:
            # forward pass the neural net
            emb = C[torch.tensor([context])] # (1, block_size, n_embd)
            h = torch.tanh(emb.view(1,-1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))

sample_model()
