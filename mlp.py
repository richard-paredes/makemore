import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
# print(itos)

BLOCK_SIZE = 3
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

import random 
random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

EMBEDDING_NDIM = 30
NUM_HIDDENLAYER = 75
BATCH_SIZE = 85
LEARNING_RATE = 0.1
DECAYED_LEARNING_RATE = 0.01
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,EMBEDDING_NDIM), generator=g) # -> (27,10), after increasing embeddings
W1 = torch.randn((EMBEDDING_NDIM*BLOCK_SIZE,NUM_HIDDENLAYER), generator=g) # -> (6,300)
b1 = torch.randn(NUM_HIDDENLAYER, generator=g) # (300)
W2 = torch.randn((NUM_HIDDENLAYER,27), generator=g) # (300,27)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
print(f'Num parameters: {sum(p.nelement() for p in parameters)}')

for p in parameters:
    p.requires_grad = True
for i in range(30000):
    lr = LEARNING_RATE if i < 25000 else DECAYED_LEARNING_RATE
    # mini batch construct
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))

    emb = C[Xtr[ix]] # only get (32,3,2)
    h = torch.tanh(emb.view(-1,EMBEDDING_NDIM*BLOCK_SIZE) @ W1 + b1)
    logits = h @ W2 + b2
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(32), Y].log().mean()

    # built-in pytorch method that evaluates the loss the same,
    # but more efficient. forward and backward pass are more efficiently in mathemtical
    # expresisons. it is more memory efficient. it also is more tolerant to extreme large numbers
    # which would otherwise cause the probabilities to explode
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -lr * p.grad
    # print(loss.item())
print(f'Training loss: {loss.item()}')

emb = C[Xdev]
h = torch.tanh(emb.view(-1,EMBEDDING_NDIM*BLOCK_SIZE) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(f'Testing loss: {loss.item()}')

# the dev loss and training loss are within teh same ballpark
# which indicates undefitting, caused by a small neural net
# scale up the neural net to increase performance

# example of scaling up neural net
# increase hidden layer 100 -> 300
# increase biases and 300 inputs to final layer

# the bigger we make the neural net, the longer it takes to converge
# mini batches causes noise in the losses when plotted on the numer of steps
# we can increase the batch size
# additionally, the embedding size could be a bottleneck to the net,
# as it is trying to stuff two much information in a vector of 2 dimension

# if you plot the embeddings for the characters,
# you will see that the model clusters characters after training
# interestingly, it has the vowel characters clustered together,
# indicating that they are somewhat similar

'''
Typical flow:

Run experiments to run different hyper parameters

Analyze which ones give the best dev set performance

Run the best model on the validation set and report that as finding in your paper
'''

for _ in range(20):
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        emb = C[torch.tensor([context])] # (1, block_size, d)
        h = torch.tanh(emb.view(1,-1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits,dim=1)
        ix = torch.multinomial(probs,num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

