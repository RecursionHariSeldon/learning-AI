import torch
import matplotlib.pyplot as plt

names = open('names.txt','r').read().splitlines()
N = torch.zeros((27,27),dtype = torch.int32)
chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars) }
stoi['.'] = 0
itos = {i:s for s,i in stoi.items() }

for name in names:
    full = ['.'] + list(name) + ['.']
    for ch1,ch2 in zip( full, full[1:] ):
        N[stoi[ch1]][stoi[ch2]] += 1
print(N)

# now the visualizer
plt.figure(figsize = (16,16) )
plt.imshow(N, cmap = 'Blues')

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='green')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='green')
plt.axis('off');