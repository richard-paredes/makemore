words = open('names.txt', 'r').read().splitlines()
print(words[:10])
print(len(words))

print("Min:", min(len(w) for w in words))
print("Max:", max(len(w) for w in words))

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        # print(ch1, ch2)

# print(b)
# print(sorted(b.items(), key = lambda kv : -kv[1]))

import torch

N = torch.zeros((27,27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))

s_to_i = {s:i+1 for i,s in enumerate(chars)}
s_to_i['.'] = 0
i_to_s = {i:s for s,i in s_to_i.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = s_to_i[ch1]
        ix2 = s_to_i[ch2]
        N[ix1, ix2] += 1

'''
import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(N,cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = i_to_s[i] = i_to_s[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i,j].item(), ha="center", va="top", color="gray")
plt.axis('off')
'''

'''
p = N[0].float()
p /= p.sum()
print(p) # convert frequencies into probabilities to sample from
'''

# Using a generator so that probabilities are deterministic every run
# g = torch.Generator().manual_seed(2147483647)
# p = torch.rand(3, generator=g)
# p /= p.sum()

# print(p)

# ix = torch.multinomial(p, num_samples=20, replacement=True, generator=g)
# print(i_to_s[ix])

N = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(stoi)
print(itos)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

g = torch.Generator().manual_seed(2147483647)

output = []
ix = 0
while True:
    p = N[ix].float()
    p = p / p.sum()
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    output.append(i_to_s[ix])
    if ix == 0:
        break
print(output)
