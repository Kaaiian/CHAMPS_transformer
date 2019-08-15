import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerBlock(nn.Module):
    def __init__(self, channels, dim=3, heads=6):
        super(TransformerBlock, self).__init__()
        self.tchannels = channels * 2
        self.dim = dim
        self.heads = heads
        self.softmax = nn.Softmax(dim=2).to(device)
        self.layernorm = nn.LayerNorm(self.tchannels)

        self.linear1 = nn.Linear(self.tchannels, self.tchannels)
        self.linear2 = nn.Linear(self.tchannels, self.tchannels)
        self.w_q = nn.Linear(self.tchannels, self.dim)
        self.w_k = nn.Linear(self.tchannels, self.dim)
        self.w_v = nn.Linear(self.tchannels, self.dim)
        self.w_o = nn.Linear(self.heads * self.dim, self.tchannels)

    def forward(self, x):
        z_list = []
        for _ in range(self.heads):
            q = self.w_q(x)
            k = self.w_k(x)
            v = self.w_v(x)
            k_t = torch.transpose(k, dim0=1, dim1=2)
            soft = self.softmax(torch.matmul(q, k_t) /
                                np.sqrt(q.shape[-1]))
            z = torch.matmul(soft, v)
            z_list.append(z)
        z = torch.cat(z_list, dim=2)
        z = self.w_o(z)
        r0 = self.linear1(z)
        r0 = x + r0
        r1 = self.layernorm(r0)
        r1 = self.linear2(r1)
        r = r0 + r1
        r = self.layernorm(r)
        return r


class TransNet(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, inner_channels):
        super(TransNet, self).__init__()
        self.channels = inner_channels

        self.elem_in = nn.Linear(14, self.channels)
        self.elem_int = nn.Linear(self.channels, self.channels)

        self.dist_in = nn.Linear(6, self.channels)
        self.dist_int = nn.Linear(self.channels, self.channels)

        self.linear_final = nn.Linear(self.channels * 29 * 2, 1)

        self.transformerblocks = nn.ModuleList([TransformerBlock(self.channels,
                                                                 dim=64,
                                                                 heads=8)
                                               for _ in range(6)])

        self.leaky = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.in1 = nn.InstanceNorm1d(self.channels)
        self.softplus2 = nn.Softplus()

    def forward(self, elem_vec, relative_pos, data_id):
        # elem_feat block
        x_elem = self.elem_in(elem_vec)
        x_elem = self.leaky(x_elem)
        x_elem = self.elem_int(x_elem)
        x_elem = self.leaky(x_elem)

        x_dist = self.dist_in(relative_pos)
        x_dist = self.leaky(x_dist)
        x_dist = self.dist_int(x_dist)
        x_dist = self.leaky(x_dist)

        x = torch.cat([x_dist, x_elem], dim=2)

        for block in self.transformerblocks:
            x = block(x)

        x = self.linear_final(x.view(-1, 29 * self.channels * 2))

        return x, data_id
