"""
Code adapted from https://github.com/MarcCoru/crop-type-mapping
Copyright (c) 2019 Marc Rußwurm
"""

import torch
import torch.nn as nn
import copy
import numpy as np


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


def get_decoder(n_neurons):
    """Returns an MLP with the layer widths specified in n_neurons.
    Every linear layer but the last one is followed by BatchNorm + ReLu

    args:
        n_neurons (list): List of int that specifies the width and length of the MLP.
    """
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        if i < (len(n_neurons) - 2):
            layers.extend([
                nn.BatchNorm1d(n_neurons[i + 1]),
                nn.ReLU()
            ])
    m = nn.Sequential(*layers)
    return m


class TempConv(nn.Module):
    """
    Temporal CNN
    """
    def __init__(self, input_size, nker, seq_len, nfc, mlp4, positions=None):
        super(TempConv, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.name = 'TempCNN_'

        self.nker = copy.deepcopy(nker)
        self.nfc = copy.deepcopy(nfc)
        self.name += '|'.join(list(map(str, self.nker)))
        self.decoder = get_decoder(mlp4)
        if self.nfc is not None:
            self.name += 'FC'
            self.name += '|'.join(list(map(str, self.nfc)))

        conv_layers = []
        self.nker.insert(0, input_size)
        for i in range(len(self.nker) - 1):
            conv_layers.extend([
                nn.Conv1d(self.nker[i], self.nker[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm1d(self.nker[i + 1]),
                nn.ReLU()
            ])
        self.conv1d = nn.Sequential(*conv_layers)

        self.nfc.insert(0, self.nker[-1] * seq_len)
        lin_layers = []
        for i in range(len(self.nfc) - 1):
            lin_layers.extend([
                nn.Linear(self.nfc[i], self.nfc[i + 1]),
                nn.BatchNorm1d(self.nfc[i + 1]),
                nn.ReLU()
            ])
        self.linear = nn.Sequential(*lin_layers)

        if positions is not None:
            self.position_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(positions, input_size, T=1000),
                freeze=True)
        else:
            self.position_enc = None

    def forward(self, input, batch_positions=None):
        sz_b, seq_len, _ = input.shape
        if self.position_enc is not None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(input.device)
            enc_output = input + self.position_enc(src_pos)
        else:
            enc_output = input

        out = self.conv1d(enc_output.permute(0, 2, 1))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = self.decoder(out)
        return out