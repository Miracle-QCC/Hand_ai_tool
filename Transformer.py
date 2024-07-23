import math

import torch
from torch import nn

class MultiAttention(nn.Module):
    def __init__(self, emb_size,head_size,expand_fraction=2):
        super(MultiAttention, self).__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.expand = expand_fraction
        self.head_embsize = emb_size // head_size * expand_fraction
        self.q = nn.Linear(emb_size,emb_size * expand_fraction)
        self.k = nn.Linear(emb_size,emb_size * expand_fraction)
        self.v = nn.Linear(emb_size,emb_size * expand_fraction)
        self.act = nn.ReLU()

    def reshape_batch(self,x):
        n,seq_len,_ = x.shape
        return x.reshape(n,seq_len,self.head_size,self.head_embsize).permute(0,2,1,3).reshape(-1,seq_len,self.head_embsize)

    def forward(self,query,key,value,mask=None):
        N,seq_len,_ = query.shape
        Q = self.act(self.q(query))
        K = self.act(self.k(key))
        V = self.act(self.v(value))

        Q = self.reshape_batch(Q)
        K = self.reshape_batch(K)
        V = self.reshape_batch(V)

        weight = Q.matmul(K.transpose(-2,-1)) / math.sqrt(self.head_embsize)
        if mask is not None:
            mask = mask.repeat(weight.shape[0],1,1)
            weight.masked_fill_(mask==0,-1e7)
        weight = torch.softmax(weight,dim=-1)

        y = weight.matmul(V)

        return y.reshape(N,self.head_size,seq_len,-1).permute(0,2,1,3).reshape(-1,seq_len,self.head_embsize*self.head_size)



if __name__ == '__main__':
    ma = MultiAttention(512,8)
    x = torch.rand((1,10,512))
    mask = torch.tril(torch.ones(10,10)).unsqueeze(0)
    ma(x,x,x,mask)
