import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


# Training set consists of two sets of data:
# xs = inputs
# ys = labels
xs, ys = [], []

# Build training set
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# How are we going to feed in the examples into a neural network? We have indices.
# One-hot encoding
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)

# x_encoding
xenc = F.one_hot(xs, num_classes=27).float() # cast as float, which can be fed to neural nets

# W = torch.randn((27,1))
logits = xenc @ W # log-counts, somewhat similar to probabilities in the scratch_bigram
counts = logits.exp() # equivalent N, where a bigram frequency is counted
prob = counts / counts.sum(1,keepdims=True)


# Step-by-step Explanation
nlls = torch.zeros(5)
for i in range(5):
    # i-th bigram:
    x = xs[i].item() # input character index
    y = ys[i].item() # label character index
    print('----')
    print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes ({x},{y}))')
    print(f'input to the neural net: {x}')
    
    print('output probabilities from neural net:', prob[i])
    print('label (actual next char):', y)
    p = prob[i,y]
    print('probability assigned by net to the correct char:', p.item())
    logp = torch.log(p)
    print('log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood:', nll.item())
    nlls[i] = nll

print('====')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())

# more efficient way of doing forward pass in the above loop
# loss = prob[torch.arange(5), ys].log().mean()

# backward pass
W.grad = None # set gradient to zero
# loss.backward()

# update net weights based on the background prop
W.data += -0.1 * W.grad

'''
xs,ys = [],[]
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('Number of examples: ', num)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)

for k in range(10):

    #forward_pass
    xenc = F.one_hot(xs, num_classes=27).float() # Input to net as one-hot-encoding
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1,keepdims=True)
    loss = -probs[torch.arange(num),ys].log().mean()
    print(f'Loss: {loss.item()}')

    #backward_pass
    W.grad = None
    loss.backward()

    #tune
    W.data += -0.1 * W.grad
'''