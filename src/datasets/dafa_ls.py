import tqdm
from src.utils.paths import DATA_PATH
import torch.utils.data as tdata
import os
from PIL import Image
import numpy as np
import torch
from datetime import datetime
import torchvision
import json
from torchvision.utils import _log_api_usage_once


FOLD_DICT = json.load(open(os.path.join(DATA_PATH, 'DAFA_LS/fold_dict.json')))
DATES = [f'{str(year)}_{month}' for year in range(2016, 2024) for month in ['01', '02', '03', '04', '05', '06',
                                                                            '07', '08', '09', '10', '11', '12']]
DATES_2023 = [f'{str(year)}_{month}' for year in [2023] for month in ['01', '02', '03', '04', '05', '06',
                                                                      '07', '08', '09', '10', '11', '12']]


class DAFA_LS(tdata.Dataset):
    def __init__(self, split='train', height=None, width=None, norm=True, pixel_set=None, semantic=False,
                 augment=True, single=False, mask_mode='none', pixel_wise=False, stats='dafa_ls'):
        super(DAFA_LS, self).__init__()

        if stats == 'dafa_ls':
            self.mean = torch.tensor([172.39825689, 149.42724701, 111.42677006])
            self.std = torch.tensor([42.36875904, 40.11172176, 42.71382535])
        elif stats == 'imagenet':
            self.mean = torch.tensor([123.675, 116.28, 103.53])
            self.std = torch.tensor([58.395, 57.12, 57.375])
        else:
            raise ValueError(f'stats has to be either `dafa_ls` or `imagenet`, not {stats}.')

        self.semantic = semantic
        self.mask_mode = mask_mode
        self.split = split
        self.height = height
        self.width = width
        self.single = single
        self.norm = norm
        self.augment = augment
        self.pixel_set = pixel_set
        self.pixel_wise = pixel_wise
        data_path_looted = os.path.join(DATA_PATH, 'DAFA_LS/looted')
        data_path_preserved = os.path.join(DATA_PATH, 'DAFA_LS/preserved')
        all_sub_dirs_looted, all_sub_dirs_preserved = [], []
        if 'test' in split or 'val' in split:
            curr_dirs = FOLD_DICT[split]
            all_sub_dirs_looted += curr_dirs['looted']
            all_sub_dirs_preserved += curr_dirs['preserved']
        else:
            curr_dirs = FOLD_DICT['train']
            all_sub_dirs_looted += curr_dirs['looted']
            all_sub_dirs_preserved += curr_dirs['preserved']
            for k in range(1, 6):
                if k != int(split[-1]):
                    curr_dirs = FOLD_DICT[f'val_{k}']
                    all_sub_dirs_looted += curr_dirs['looted']
                    all_sub_dirs_preserved += curr_dirs['preserved']
        all_sub_dirs_looted.sort()
        all_sub_dirs_preserved.sort()
        sub_dirs, labels, positions = [], [], []
        if not self.single:
            for item in all_sub_dirs_looted:
                sub_dirs.append(os.path.join(data_path_looted, str(item)))
                labels.append(1)
            self.num_looted = len(sub_dirs)
            for item in all_sub_dirs_preserved:
                sub_dirs.append(os.path.join(data_path_preserved, str(item)))
                labels.append(0)
            self.num_preserved = len(sub_dirs) - self.num_looted
        else:
            curr_dates = DATES_2023
            for item in all_sub_dirs_looted:
                for date_id, date in enumerate(curr_dates):
                    if os.path.exists(os.path.join(data_path_looted, str(item), f'{date}.jpg')):
                        sub_dirs.append(os.path.join(data_path_looted, str(item), f'{date}.jpg'))
                        labels.append(1)
                        positions.append(date_id)
            self.num_looted = len(sub_dirs)
            for item in all_sub_dirs_preserved:
                for date_id, date in enumerate(curr_dates):
                    if os.path.exists(os.path.join(data_path_preserved, str(item), f'{date}.jpg')):
                        sub_dirs.append(os.path.join(data_path_preserved, str(item), f'{date}.jpg'))
                        labels.append(0)
                        positions.append(date_id)
            self.num_preserved = len(sub_dirs) - self.num_looted
        self.sub_dirs = sub_dirs
        self.labels = labels
        self.positions = positions
        self.len = len(self.sub_dirs)

        if self.pixel_wise and 'train' in self.split:
            print("Preparing pixel-wise data...")
            data, label = None, None
            for item in tqdm.tqdm(range(self.len)):
                mask = (torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], 'mask.png')))) / 255).flatten()
                curr_label = [self.labels[item] for _ in range((mask == 1).sum())]
                empty_list = [[0 for _ in range((mask == 1).sum())] for _ in range(3)]
                curr_data = []
                for date_id, date in enumerate(DATES):
                    if os.path.exists(os.path.join(self.sub_dirs[item], f'{date}.jpg')):
                        curr_data_date = torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], f'{date}.jpg'))))
                        curr_data_date = curr_data_date.permute(2, 0, 1).flatten(1)
                        curr_data_date = curr_data_date[:, mask == 1]
                        curr_data.append(curr_data_date.to(torch.uint8).numpy().tolist())
                    else:
                        curr_data.append(empty_list)
                if data is None:
                    data = torch.tensor(curr_data, dtype=torch.uint8).permute(2, 1, 0)
                    label = torch.tensor(curr_label, dtype=torch.uint8)
                else:
                    data = torch.cat([data, torch.tensor(curr_data, dtype=torch.uint8).permute(2, 1, 0)])
                    label = torch.cat([label, torch.tensor(curr_label, dtype=torch.uint8)])
            self.data = data
            self.labels = label
            self.len = self.data.shape[0]
        print(f'Dataset len: {self.len}')
        self.randomVFlip = torchvision.transforms.RandomVerticalFlip(p=0.5)
        self.randomHFlip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.randomRotate = RandomRotation90()
        self.randomResizeCrop = torchvision.transforms.RandomResizedCrop(size=(self.height, self.width),
                                                                         scale=(0.3, 1.))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        data, positions = [], []
        if self.pixel_wise:
            if 'train' in self.split:
                data = self.data[item].float().permute(1, 0)
                data = (data - self.mean) / self.std
                label = torch.tensor([self.labels[item]]).to(torch.int64)
                return data, label
            else:
                for date_id, date in enumerate(DATES):
                    if os.path.exists(os.path.join(self.sub_dirs[item], f'{date}.jpg')):
                        data.append(torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], f'{date}.jpg')))))
                    else:
                        data.append(torch.zeros((266, 266, 3)))
                data = torch.stack(data, dim=0).permute(0, 3, 1, 2).flatten(2)
                mask = (torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], 'mask.png')))) / 255).flatten()
                data = data[:, :, mask == 1]
                if self.norm:
                    data = (data - self.mean[:, None]) / self.std[:, None]
                data = data.permute(2, 0, 1)
                label = torch.tensor([self.labels[item]])
                return data, label

        elif self.pixel_set is not None:
            for date_id, date in enumerate(DATES):
                if os.path.exists(os.path.join(self.sub_dirs[item], f'{date}.jpg')):
                    data.append(torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], f'{date}.jpg')))))
                    positions.append(date_id)
            data = torch.stack(data, dim=0).permute(0, 3, 1, 2).flatten(2)
            positions = torch.tensor(positions)
            mask = (torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], 'mask.png')))) / 255).flatten()
            data = data[:, :, mask == 1]
            if data.shape[-1] > self.pixel_set:
                idx = np.random.choice(list(range(data.shape[-1])), size=self.pixel_set, replace=False)
                data = data[:, :, idx]
                mask = torch.ones(self.pixel_set)
            elif data.shape[-1] < self.pixel_set:
                if data.shape[-1] == 0:
                    data = torch.zeros((*data.shape[:2], self.pixel_set))
                    mask = torch.zeros(self.pixel_set)
                    mask[0] = 1
                else:
                    x = torch.zeros((*data.shape[:2], self.pixel_set))
                    x[:, :, :data.shape[-1]] = data
                    x[:, :, data.shape[-1]:] = torch.stack([x[:, :, 0] for _ in range(data.shape[-1], x.shape[-1])], dim=-1)
                    data = x
                    mask = torch.tensor(
                        [1 for _ in range(data.shape[-1])] + [0 for _ in range(data.shape[-1], self.pixel_set)])
            else:
                mask = torch.ones(self.pixel_set)
            if self.norm:
                data = (data - self.mean[:, None]) / self.std[:, None]
            label = torch.tensor([self.labels[item]])
            return data, positions, mask.int(), label

        else:
            if self.single:
                data = torch.tensor(np.array(Image.open(self.sub_dirs[item]))).permute(2, 0, 1)
                mask = torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item][:-12], 'mask.png')))) / 255
                positions = torch.tensor([self.positions[item]])
            else:
                if 'train' in self.split:
                    kept_date = []
                    for date_id, date in enumerate(DATES):
                        if os.path.exists(os.path.join(self.sub_dirs[item], f'{date}.jpg')):
                            kept_date.append(date_id)
                    random_dates = [np.random.choice(range(len(kept_date) // 8), 3, replace=False) + k*(len(kept_date) // 8)*np.array([1, 1, 1]) for k in range(8)]
                    random_dates = np.concatenate(random_dates)
                    random_dates.sort()
                    kept_date = [kept_date[k] for k in random_dates]
                    for date_id in kept_date:
                        data.append(torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], f'{DATES[date_id]}.jpg')))))
                        positions.append(date_id)
                else:
                    for date_id, date in enumerate(DATES):
                        if os.path.exists(os.path.join(self.sub_dirs[item], f'{date}.jpg')):
                            data.append(torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], f'{date}.jpg')))))
                            positions.append(date_id)
                data = torch.stack(data, dim=0).permute(0, 3, 1, 2)
                positions = torch.tensor(positions)
                mask = torch.tensor(np.array(Image.open(os.path.join(self.sub_dirs[item], 'mask.png')))) / 255
            data = self.resize(data)
            mask = self.resize(mask[None]).squeeze()
            label = torch.tensor([self.labels[item]])
            if self.norm:
                data = (data - self.mean[:, None, None]) / self.std[:, None, None]
            if 'train' in self.split and self.augment:
                if not self.single:
                    augmented = self.random_augment(torch.cat([data, mask[None].expand(3, -1, -1)[None]], dim=0))
                else:
                    augmented = self.random_augment(torch.cat([data[None], mask[None].expand(3, -1, -1)[None]], dim=0))
                data, mask = augmented[:-1].squeeze(), augmented[-1][0].squeeze()
            if self.mask_mode == 'channel':
                if not self.single:
                    data = torch.cat([data, mask[None, None].expand(data.shape[0], -1, -1, -1)], dim=1)
                else:
                    data = torch.cat([data, mask[None]], dim=0)
            elif self.mask_mode == 'multiply':
                if not self.single:
                    data = data * mask[None, None]
                else:
                    data = data * mask[None]
            else:
                assert self.mask_mode == 'none', f'`mask_mode` has to be either `none`, `channel` or `multiply` not {self.mask_mode}'
            if self.semantic:
                label = torch.where(mask == 1, label + 1, 0)
            return data, positions, mask, label

    def resize(self, data):
        if self.height is not None and self.width is not None:
            data = torchvision.transforms.functional.resize(data, (self.height, self.width))
        return data

    def random_resize_crop(self, data):
        return self.randomResizeCrop(data)

    def random_flipv(self, data):
        return self.randomVFlip(data)

    def random_fliph(self, data):
        return self.randomHFlip(data)

    def random_rotate(self, data):
        return self.randomRotate(data)

    def random_augment(self, data):
        data = self.random_resize_crop(data)
        data = self.random_fliph(data)
        data = self.random_flipv(data)
        data = self.random_rotate(data)
        return data


def calculate_index(year, month):
    input_date = datetime(year, month, 1)
    base_date = datetime(2015, 12, 1)
    month_difference = (input_date.year - base_date.year) * 12 + input_date.month - base_date.month
    return month_difference


def retrieve_date(index):
    index = int(index)
    base_date = datetime(2015, 12, 1)
    month = (index + base_date.month) % 12
    year = (index + base_date.month) // 12 + base_date.year
    return month, year


class RandomRotation90(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    @staticmethod
    def get_params():
        angle = float([0, 90, 180, 270][np.random.randint(4)])
        return angle

    def forward(self, img):
        channels, _, _ = torchvision.transforms.functional.get_dimensions(img)
        angle = self.get_params()
        return torchvision.transforms.functional.rotate(img, angle)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"("
        format_string += ")"
        return format_string
