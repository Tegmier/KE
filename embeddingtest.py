import torch
import torch.nn as nn

emb = nn.Embedding(3,4)
emb(torch.tensor([0],dtype=torch.int64))
print(emb(torch.tensor([0],dtype=torch.int64)))

emb_weight = torch.rand(3,5, requires_grad=True)
emb2 = nn.Embedding.from_pretrained(emb_weight)
print(emb2(torch.tensor([0], dtype=torch.int64)))