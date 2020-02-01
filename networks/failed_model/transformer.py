import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class SelfAttention(nn.Module):
    ''' SelfAttention module '''

    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
    
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k)
        self.w_ks = nn.Linear(d_model, d_k)
        self.w_vs = nn.Linear(d_model, d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_v, d_model)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input):
        
        q = enc_input
        k = enc_input
        v = enc_input

        residual = q

        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        output = self.attention(q, k, v)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


def main():
    d_model = 365
    d_k = 64
    d_v = 64

    model = SelfAttention(d_model, d_k, d_v)
    enc_input = torch.rand((4, 8, 365))
    
    output = model(enc_input)
    print(enc_input.size(), output.size())

if __name__ == '__main__':
    main()
