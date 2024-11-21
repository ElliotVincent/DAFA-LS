import torch
import collections.abc
import re
from torch.nn import functional as F
import json
import os
from pathlib import Path
from src.model.backbone_ltae import BackboneLTAE
from src.datasets.dafa_ls import DAFA_LS
from src.model.resnet import resnet20, resnet18, resnet34
from src.model.tempcnn import TempConv
from src.model.utae import UTAE
from src.model.tsvit import TSViT
from src.model.transformer import TransformerEncoder
from src.model.duplo import DuPLO
from src.model.dofa import DOFA
from src.model.scalemae import vit_large_patch16 as ScaleMAE
from src.model.satmae import vit_large_patch16 as SatMAE
from src.utils.paths import PROJECT_PATH


def get_model(config):
    name = config['model']['name']
    input_dim = config['dataset'].get('input_dim', 3)
    num_classes = 2

    if name == 'resnet20':
        model = resnet20()
        if config['model'].get('pretrained', False):
            state_dict = torch.load(os.path.join(PROJECT_PATH, 'weights/resnet20-12fca82f.th'))['state_dict']
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        if config['model'].get('frozen', True):
            for param in model.parameters():
                param.requires_grad = False
        model.linear = torch.nn.Linear(64, 2)
        for param in model.linear.parameters():
            param.requires_grad = True

    elif name == 'resnet18':
        if config['model'].get('pretrained', False):
            model = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet18()
        if config['model'].get('frozen', True):
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(512, 2)
        for param in model.fc.parameters():
            param.requires_grad = True

    elif name == 'resnet34':
        if config['model'].get('pretrained', False):
            model = resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = resnet34()
        if config['model'].get('frozen', True):
            for param in model.parameters():
                param.requires_grad = False
        model.fc = torch.nn.Linear(512, 2)
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif name == 'dofa':
        model = DOFA(pretrained=True, is_head=True,
                     output_dim=2, path=os.path.join(PROJECT_PATH, 'weights/DOFA_ViT_base_e100.pth'),
                     wavelengths=[0.665, 0.56, 0.49], modalities=[0, 1, 2])
        if config['model'].get('frozen', True):
            for param in model.parameters():
                param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    elif name == 'satmae':
        model = SatMAE()
        model.head = None
        pretrained = config['model'].get('pretrained', True)
        frozen = config['model'].get('frozen', True)
        if pretrained:
            path = os.path.join(PROJECT_PATH, 'weights/fmow_pretrain.pth')
            state_dict = torch.load(os.path.join(path))['model']
            new_state_dict = {}
            for key, val in state_dict.items():
                if not 'decoder' in key and key != 'mask_token':
                    new_state_dict[key] = val
            model.load_state_dict(new_state_dict)
        model.head = torch.nn.Linear(1024, 2)
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    elif name == 'scalemae':
        model = ScaleMAE()
        model.head = None
        pretrained = config['model'].get('pretrained', True)
        frozen = config['model'].get('frozen', True)
        if pretrained:
            path = os.path.join(PROJECT_PATH, 'weights/scalemae-vitlarge-800.pth')
            state_dict = torch.load(os.path.join(path))['model']
            new_state_dict = {}
            for key, val in state_dict.items():
                if not 'fpn.fpn' in key and not 'fcn_high' in key and not 'fcn_low' in key and not 'decoder' in key and key != 'mask_token':
                    new_state_dict[key] = val
            model.load_state_dict(new_state_dict)
        model.head = torch.nn.Linear(1024, 2)
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    
    elif name == 'duplo':
        model = DuPLO(input_dim=config['model']['input_dim'],
                      nclasses=config['model']['nclasses'],
                      sequencelength=config['model']['sequencelength'],
                      dropout=config['model']['dropout'])

    elif name == 'tsvit':
        dico = config['model']
        dico.pop('name')
        model = TSViT(dico, mode=config['model'].get('mode', 'cls'))
        config['model']['name'] = 'tsvit'

    elif name == 'utae':
        model = UTAE(input_dim=config['model']['input_dim'],
                     encoder_widths=config['model']['encoder_widths'],
                     decoder_widths=config['model']['decoder_widths'],
                     out_conv=config['model']['out_conv'],
                     str_conv_k=config['model']['str_conv_k'],
                     str_conv_s=config['model']['str_conv_s'],
                     str_conv_p=config['model']['str_conv_p'],
                     agg_mode=config['model']['agg_mode'],
                     encoder_norm=config['model']['encoder_norm'],
                     n_head=config['model']['n_head'],
                     d_model=config['model']['d_model'],
                     d_k=config['model']['d_k'],
                     encoder=config['model']['encoder'],
                     return_maps=config['model']['return_maps'],
                     pad_value=config['model']['pad_value'],
                     padding_mode=config['model']['padding_mode'])

    elif name == 'transformer':
        model = TransformerEncoder(in_channels=config['model']['in_channels'],
                                   len_max_seq=config['model']['len_max_seq'],
                                   d_word_vec=config['model']['d_word_vec'], d_model=config['model']['d_model'],
                                   d_inner=config['model']['d_inner'], n_layers=config['model']['n_layers'],
                                   n_head=config['model']['n_head'], d_k=config['model']['d_k'],
                                   d_v=config['model']['d_v'],
                                   dropout=config['model']['dropout'], nclasses=config['model']['nclasses']
                                   )

    elif name == 'tempcnn':
        model = TempConv(input_size=config['model']['in_channels'],
                         nker=config['model']['nker'],
                         nfc=config['model']['nfc'],
                         seq_len=config['model']['seq_len'],
                         mlp4=config['model']['mlp4'])

    elif name == 'backboneltae':
        in_channels = config['model']['in_channels']
        backbone = config['model']['backbone']
        model = BackboneLTAE(input_dim=input_dim, num_classes=num_classes, in_channels=in_channels, n_head=8,
                             d_k=4, mlp=[in_channels, in_channels // 2], dropout=0.2, d_model=in_channels, T=300,
                             return_att=False, positional_encoding=True, backbone=backbone,
                             frozen=config['model'].get('frozen', True),
                             pretrained=config['model'].get('pretrained', True))

    else:
        available = '[`utae`, `backboneltae`, `resnet20`, `dofa`, `tempcnn`, `duplo`, `tsvit`, `transformer`]'
        raise ValueError(f'Model name has to be in {available} but got `{name}` instead.')
    return model


def get_loader(config, split, fold, augment=True, batch_size=None, single=None):
    img_h = config['dataset'].get('height', None)
    img_w = config['dataset'].get('width', None)
    single = config['dataset'].get('single', False) if single is None else single
    mask_mode = config['dataset'].get('mask_mode', 'none')
    pixel_set = config['dataset'].get('pixel_set', None)
    pixel_wise = config['dataset'].get('pixel_wise', False)
    semantic = config['dataset'].get('semantic', False)
    stats = config['dataset'].get('stats', 'dafa_ls')
    pixel_set = None if pixel_set == 'None' else pixel_set
    dataset = DAFA_LS(split=f'{split}_{fold}' if split != 'test' else split, height=img_h, width=img_w,
                      mask_mode=mask_mode, augment=augment, single=single, pixel_set=pixel_set,
                      pixel_wise=pixel_wise, semantic=semantic, stats=stats)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['training'].get('batch_size', 4) if batch_size is None else batch_size,
            shuffle=True if (split == 'train' and augment) else False,
            drop_last=True if (split == 'train' and augment) else False,
            collate_fn=pad_collate,
            num_workers=config['training'].get('num_workers', 8)
    )
    return loader


def get_criterion(config, device):
    name = config['training'].get('criterion', 'ce')
    weight = config['training'].get('weight', False)
    ignore_index = config['training'].get('ignore_index', -1)
    if name == 'ce':
        if not weight:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=device))
    elif name == 'bce':
        if not weight:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([0.2, 0.8], device=device))
    else:
        raise ValueError(f'Criterion should be one of [`ce`, `bce`] not {name}.')
    return criterion


def coerce_to_path_and_check_exist(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path


def coerce_to_path_and_create_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).view(-1, m, *list(elem.size())[1:])
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


class Logger:
    def __init__(
            self,
            split,
            name,
            metrics=None,
    ):
        super(Logger, self).__init__()
        self.split = split
        self.name = name
        if metrics is None:
            self.metrics = {'loss': {},
                            'acc': {},
                            'mean_acc': {},
                            'fpr': {},
                            'auroc': {},
                            'f1': {},
                            'precision': {},
                            'recall': {},
                            'conf_mat': {},
                            'class_acc': {},
                            'lr': {}
                            }
        else:
            self.metrics = metrics

    def update(self, viz, n_iter, res_dir, metrics):
        for key, vals in metrics.items():
            self.metrics[key][n_iter] = vals
        with open(os.path.join(res_dir, f'{self.split}_{self.name}.json'), 'w') as file:
            file.write(json.dumps(self.metrics, indent=4))
        if viz is not None:
            update_viz(viz, metrics, self.split, n_iter)


def update_viz(viz, metrics, mode, n_iter):
    viz.plot('loss', mode, 'Loss', n_iter, metrics['loss'])
    viz.plot('acc', mode, 'Overall Accuracy', n_iter, metrics['acc'])
    viz.plot('m_acc', mode, 'Mean Accuracy', n_iter, metrics['mean_acc'])
