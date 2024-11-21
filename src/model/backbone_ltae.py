"""
Code adapted from https://github.com/VSainteuf/utae-paps
Copyright (c) 2021 Vivien Sainte Fare Garnot
"""

import os
import torch
import torch.nn as nn
import copy
import torchvision
from src.model.ltae import MultiHeadAttention
from src.model.positional_encoding import PositionalEncoder
from src.model.dofa import DOFA
from src.model.scalemae import vit_large_patch16 as ScaleMAE
from src.model.satmae import vit_large_patch16 as SatMAE
from src.model.pse import PixelSetEncoder
from src.utils.paths import PROJECT_PATH


class BackboneLTAE(nn.Module):
    def __init__(
        self,
        input_dim=3,
        num_classes=2,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
        backbone='pse',
        pretrained=True,
        frozen=True
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
            backbone (str): What backbone to use
            pretrained (bool): Whether the backbone is loaded with pretrained weights
            frozen (bool): Whether the backbone is frozen during training
        """
        super(BackboneLTAE, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head
        self.backbone = backbone

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        if backbone == 'mlp':
            self.pixel_encoder = torch.nn.Linear(in_features=input_dim, out_features=in_channels)

        elif backbone == 'pse':
            self.img_encoder = PixelSetEncoder(input_dim=input_dim,
                                               mlp1=[input_dim, in_channels//4, in_channels//2],
                                               mlp2=[in_channels],
                                               with_extra=False,
                                               extra_size=0)

        elif backbone == 'vit_b_16':
            if pretrained:
                self.img_encoder = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
            else:
                self.img_encoder = torchvision.models.vit_b_16()
            if frozen:
                for param in self.img_encoder.parameters():
                    param.requires_grad = False
            self.img_encoder.heads.head = torch.nn.Identity()

        elif backbone == 'resnet18':
            if pretrained:
                self.img_encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.img_encoder = torchvision.models.resnet18()
            if frozen:
                for param in self.img_encoder.parameters():
                    param.requires_grad = False
            self.img_encoder.fc = torch.nn.Identity()

        elif backbone == 'dofa':
            self.img_encoder = DOFA(pretrained=pretrained, is_head=False, output_dim=2,
                         path=os.path.join(PROJECT_PATH, 'weights/DOFA_ViT_base_e100.pth'),
                         wavelengths=[0.665, 0.56, 0.49], modalities=[0, 1, 2])
            if frozen:
                for param in self.img_encoder.parameters():
                    param.requires_grad = False

        elif backbone == 'scalemae':
            self.img_encoder = ScaleMAE()
            self.img_encoder.head = None
            if pretrained:
                path = os.path.join(PROJECT_PATH, 'weights/scalemae-vitlarge-800.pth')
                state_dict = torch.load(os.path.join(path))['model']
                new_state_dict = {}
                for key, val in state_dict.items():
                    if not 'fpn.fpn' in key and not 'fcn_high' in key and not 'fcn_low' in key and not 'decoder' in key and key != 'mask_token':
                        new_state_dict[key] = val
                self.img_encoder.load_state_dict(new_state_dict)
            if frozen:
                for param in self.img_encoder.parameters():
                    param.requires_grad = False

        elif backbone == 'satmae':
            self.img_encoder = SatMAE()
            self.img_encoder.head = None
            if pretrained:
                path = os.path.join(PROJECT_PATH, 'weights/fmow_pretrain.pth')
                state_dict = torch.load(os.path.join(path))['model']
                new_state_dict = {}
                for key, val in state_dict.items():
                    if not 'decoder' in key and key != 'mask_token':
                        new_state_dict[key] = val
                self.img_encoder.load_state_dict(new_state_dict)
            if frozen:
                for param in self.img_encoder.parameters():
                    param.requires_grad = False

        mlp_decoder = [mlp[-1],  32, num_classes]
        self.decoder = get_decoder(mlp_decoder)

    def forward(self, x, batch_positions=None, mask=None, pad_mask=None):
        if self.backbone == 'mlp':
            sz_b, seq_len, d = x.shape
        elif self.backbone != 'pse':
            sz_b, seq_len, d, h, w = x.shape
        else:
            sz_b, seq_len, d, num_pix = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        if self.backbone == 'mlp':
            out = self.pixel_encoder(x)

        elif self.backbone == 'pse':
            out = self.img_encoder((x, mask))

        elif self.backbone in  ['dofa', 'satmae', 'scalemae']:
            x = x.reshape(sz_b * seq_len, d, h, w)
            out = self.img_encoder(x)
            out = out.view(sz_b, seq_len, -1)

        elif self.backbone in ['vit_b_16', 'resnet18']:
            x = x.reshape(sz_b * seq_len, d, h, w)
            out = self.img_encoder(x)
            print(out.shape)
            out = out.view(sz_b, seq_len, -1)

        else:
            raise ValueError(f'backbone should be in [`mlp`, `pse`, `dofa`, `vit_b_16`, `resnet18`, `satmae`, `scalemae`] not {self.backbone}')

        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            out = out + self.positional_encoder(batch_positions)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out

        attn = attn.view(self.n_head, sz_b, seq_len)  # head x b x t

        if self.return_att:
            return self.decoder(out), attn
        else:
            return self.decoder(out)


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
