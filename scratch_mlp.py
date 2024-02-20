import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
print(words[:8])
len(words)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

block_size = 3 # Defining context length, i.e. num chars taken to predict the next one
X, Y = [], []
for w in words:
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append
        # print(X)
        # print(Y)
        # print(context)

# builds the 32 examples (using the first 5 words) in X
# each with their corresponding label in Y
X, Y = torch.tensor(X), torch.tensor(Y)

# Now to build a NN that takes the input X to predict Y
C = torch.randn((27, 2)) # each one of 27 chars will have a 2 dim embedding

# Index an embedding table to get the input
C[5]

# alternative way of retrieving an input to the neural net using one-hot encoding (used in bigram)
# F.one_hot(torch.tensor(5), num_classes=27).float() @ C

# pytorch lets us index using lists to get more than one element back
print(C[[5,6,7]])
print(C[torch.tensor([5,6,7])]) 
print(C[torch.tensor([5,6,7,7,7])])

print(C[X].shape) # for every 32 x 3 integers, the associated embedding vector is there
print(C[X[13,2]], C[X][13,2]) # these are equal

# input layer
emb = C[X]
NUM_NUERONS = 100
W1 = torch.randn((6,NUM_NUERONS))
b1 = torch.randn(100)

# emb @ W1 + b1 # cannot matrix mult here, since dimensions aren't compatible
 
# print(torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]], 1).shape)

# print(len(torch.unbind(emb, 1)))
# torch.cat(torch.unbind(emb,1), 1)

# a = torch.arange(18)
# print(a.storage())
'''
a tensor in Pytorch always has an underlying storage as 1 dimensional numbers
which is how it is stored in computer memory

there are internals in a tensor that dictate how it's interpreted to be a 
n-dimensional vector.
we can use the tensor.view function to manipulate those internals, 
which will efficiently change the dimensions of the vector without copying any
vectors or data
'''

# print(emb.view(32,6) == torch.cat(torch.unbind(emb,1),1))

# h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
# print(h, '\n', h.shape)

# output layer
# W2 = torch.randn((100,27))
# b2 = torch.randn(27)
# logits = h @ W2 + b2
# print(logits, '\n', logits.shape)

# counts = logits.exp()
# prob = counts / counts.sum(1, keepdims=True)

# loss = -prob[torch.arange(32), Y].log().mean()
# print(loss)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)
W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100,27), generator=g)
b2 = torch.randn(27, generator=g)
LEARNING_RATE = 0.1
'''
# how do we determine a good learning rate?
# a high one will cause the loss to yoyo and be unstable
# a low one will cause the loss to never go down

# creates 1000 numbers between 1 and 0.001
torch.linspace(0.001,1,1000)
lre = torch.linspace(-3,0,1000)
lrs = 10**lre
print(lrs) # candidate learning rates to try out

# now we need to keep track of the losses that resulted from 
# using the different learning rates
lri, lossi = [], []
parameters = [C, W1, b1, W2, b2]

print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
for i in range(1000):

    # mini batch construct
    ix = torch.randint(0, X.shape[0], (32,))

    emb = C[X[ix]] # only get (32,3,2)
    h = torch.tanh(emb.view(-1,6) @ W1 + b1)
    logits = h @ W2 + b2
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(32), Y].log().mean()

    # built-in pytorch method that evaluates the loss the same,
    # but more efficient. forward and backward pass are more efficiently in mathemtical
    # expresisons. it is more memory efficient. it also is more tolerant to extreme large numbers
    # which would otherwise cause the probabilities to explode
    loss = F.cross_entropy(logits, Y[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = lrs[i]
    for p in parameters:
        p.data += -lr * p.grad
    print(loss.item())
    lri.append(lre[i])
    lossi.append(loss.item())

# after running, we can plot the graph of learning rates against lossi
# we will see that a good rate is between 0.0 and 0.2 where lossi
# is not unstable
'''
'''
after determining a good learning rate using the above strategy of 
plotting the learning rate exponents and the loss, we can stick with 
a 'good' learning rate where the loss was pretty low

after running multiple iterations, we can then do 'learning rate decay'
where we decrease the learning rate at the late stage of training to go slower

'''

'''
One thing to be cautious of is that a low loss does not imply a great model.
It can be overfitted, where it is optimized against your training data.
This means that if you try introducing new examples to it, the loss on the
predictions of those new values can be very high, which would indicate
a poor model.

This is why the concept of training data splits exist:
- training split: 80%, used to optimize parameters of model
- dev/validation split: 10%, used for development of all hyperparameters of model. e.g. size of embedding, hidden layers
- test split: 10%

You need to rarely test your model on the test split, otherwise you will just overfit
'''

