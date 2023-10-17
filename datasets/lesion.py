import os
import json

import cv2
import numpy as np
import torch
from PIL import Image
from collections import namedtuple

from torch.utils.data import Dataset


class lesionDataset(Dataset):
    # For mask images
    LesionClass = namedtuple('LesionClass', ['name', 'id', 'train_id', 'color'])
    classes = [
        LesionClass('background', 0, 4, (0, 0, 0)),
        LesionClass('Normal Tissue', 1, 0, (150, 0, 0)),
        LesionClass('invasive Cancer', 2, 1, (190, 153, 153)),
        LesionClass('Tumor', 3, 2, (128, 64, 128)),
        LesionClass('DCIS noncomedo', 4, 3, (70, 70, 70)),
        LesionClass('DCIS comedo', 5, 3, (70, 70, 70)),
        LesionClass('LCIS', 6, 3, (70, 70, 70)),
        # LesionClass('DCIS comedo', 5, 4, (0, 0, 142)),
        # LesionClass('LCIS', 6, 4, (119, 11, 32)),
    ]

    train_id_to_color = [c.color for c in classes]
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, cfg, transform, mode='cls', task='train'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.task = task
        self.transform = transform
        # Load the json file including all the information of data
        self.data_dict = self.load_data(cfg, mode, task)
        self.blank_color = 245 / 255.
        self.in_channel = cfg.inchannel
        if mode == 'tiscls' or mode == 'total_cls':
            self.num_tis_classes = cfg.model_tiscls.num_tis_classes
        if mode == 'patch_cls':
            self.max_size = int(np.ceil(3800.0 / self.cfg.patchsize) * self.cfg.patchsize)
            self.num_patches = (self.max_size / self.cfg.patchsize) ** 2
        else:
            self.max_size = 3800  # This is obtained by checking all data in a dataset.

    def get_num_patches(self):
        assert self.cfg.mode == 'patch_cls'
        return self.num_patches

    @staticmethod
    def load_data(cfg, mode, task):
        with open(cfg.dataset.json_path, 'r') as f:
            data = json.load(f)
        assert task in list(data.keys()), 'There is no {} data'.format(task)
        if mode == 'cls' or mode == 'total_cls' or mode == 'dragon' or mode == 'patch_cls':
            data = data[task]
        elif mode == 'tiscls':
            data = data[task + '_mask']
        else:
            raise NotImplementedError(f"Mode {mode} Not implemented.")
        return data

    @staticmethod
    def one_hot_encoding(class_ids: list, num_classes):
        ohe = np.zeros([1, num_classes])
        for class_id in class_ids:
            ohe[0, class_id] = 1
        return ohe

    def __getitem__(self, item):
        data = self.data_dict[list(self.data_dict.keys())[item]]
        #
        table_x = torch.from_numpy(np.array(data['x']))
        table_y = torch.from_numpy(np.array(data['y']))
        table_t = torch.from_numpy(np.array(data['t']))
        onehot_t = torch.from_numpy(lesionDataset.one_hot_encoding([table_t.numpy().item()], 2))
        one_hot_tis = None
        img = None
        mask = None
        if self.mode == 'dragon':
            return img, mask, table_x, table_y, table_t, onehot_t, None, self.mode
        else:  # For mode == 'tiscls' or mode == 'total_cls' or mode == 'cls'
            # Load a image
            img = self.load_image(data['img_path']) / 255.
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))
            # Crop the loaded image
            img = img[:, :, :self.max_size] if img.shape[2] > self.max_size else img
            img = img[:, :self.max_size, :] if img.shape[1] > self.max_size else img
            # Pad the cropped image
            img = self.padding_to_fixedsize(img, fixed_size=self.max_size, color=self.blank_color)
            img = torch.from_numpy(img)
            if self.mode == 'cls':
                return img, None, table_x, table_y, table_t, onehot_t, None, self.mode
            if self.mode == 'patch_cls':
                patches = []
                patch_s = self.cfg.patchsize
                for row in range(int(self.max_size / self.cfg.patchsize)):
                    for col in range(int(self.max_size / self.cfg.patchsize)):
                        patches.append(img[:, row * patch_s:(row + 1) * patch_s, col * patch_s:(col + 1) * patch_s])
                return torch.stack(patches, dim=1), None, table_x, table_y, table_t, onehot_t, None, self.mode
            #
            if data['mask_path'] != '-':
                mask = self.encode_target(self.load_mask(data['mask_path']))
                mask = self.padding_to_fixedsize(np.expand_dims(mask, axis=0), fixed_size=self.max_size,
                                                 color=self.blank_color)
                mask = torch.from_numpy(mask).type(torch.int) if mask is not None else mask
                #
                mask_orig = mask
                mask_per_class = []
                for class_id in range(self.cfg.model_tiscls.num_tis_classes):
                    temp = torch.zeros(1, mask.shape[1], mask.shape[2])
                    temp[mask == class_id] = 1
                    mask_per_class.append(temp)
                mask = torch.cat(mask_per_class, dim=0)
                #
                cls_id_list = list(torch.unique(mask_orig).numpy())
                try:
                    cls_id_list.remove(self.num_tis_classes)
                except:
                    pass
                one_hot_tis = torch.from_numpy(self.one_hot_encoding(cls_id_list, self.num_tis_classes))
            else:
                mask = torch.ones(self.num_tis_classes, img.shape[1], img.shape[2]) * -1
                one_hot_tis = torch.ones(1, self.num_tis_classes) * -1
            return img, mask, table_x, table_y, table_t, onehot_t, one_hot_tis, self.mode

    # For debugging
    # a = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # b = (self.decode_target(mask.numpy())).astype(np.uint8)
    # plt.imshow(np.concatenate([a, b], axis=1))

    @staticmethod
    def make_patches(img, mask, img_size, num_patches):
        img_patches = []
        mask_patches = []
        for r in range(num_patches):
            for c in range(num_patches):
                img_patches.append(
                    img[:, r * img_size: (r + 1) * img_size, c * img_size:(c + 1) * img_size].unsqueeze(dim=0))
                mask_patches.append(
                    mask[:, r * img_size: (r + 1) * img_size, c * img_size:(c + 1) * img_size].unsqueeze(dim=0))
        return torch.cat(img_patches, dim=0), torch.cat(mask_patches, dim=0)

    def load_image(self, img_path):
        if self.in_channel == 1:
            im = np.expand_dims(np.asarray(Image.open(img_path).convert('L')), axis=-1)
            assert im is not None, f"opencv cannot read image correctly or {img_path} not exists"
        else:
            im = cv2.cvtColor(np.asarray(Image.open(img_path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f"opencv cannot read image correctly or {img_path} not exists"
        return im

    @staticmethod
    def load_mask(mask_path):
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f"opencv cannot read image correctly or {mask_path} not exists"
        except:
            mask = np.asarray(Image.open(mask_path))
            assert mask is not None, f"Image Not Found {mask_path}, workdir: {os.getcwd()}"
        return mask

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        return cls.train_id_to_color[target]

    def __len__(self):
        return len(self.data_dict)

    @staticmethod
    def collate_fn(batch):
        # assert len(batch) == 1, 'the case of batch_size > 1 is not implemented...'
        img, mask, table_x, table_y, table_t, one_hot, one_hot_tis, mode = zip(*batch)
        #
        if mode[0] == 'demo' and len(img) > 1:
            raise ValueError('For demo, batchsize should be 1...')
        #
        img_, mask_, x_, y_, t_, one_hot_, one_hot_tis_ = [], [], [], [], [], [], []
        #
        for i in range(len(img)):
            x_.append(table_x[i].float().unsqueeze(dim=0))
            y_.append(table_y[i].float().unsqueeze(dim=0))
            t_.append(table_t[i].float().unsqueeze(dim=0))
            #
            if mode[0] != 'dragon':
                img_.append(img[i].unsqueeze(dim=0).float())
                one_hot_.append(one_hot[i])
                if mode[0] == 'tiscls' or mode[0] == 'total_cls':
                    one_hot_tis_.append(one_hot_tis[i])
                    mask_.append(mask[i].unsqueeze(dim=0).float())

        x = torch.cat(x_, dim=0)
        y = torch.cat(y_, dim=0)
        t = torch.cat(t_, dim=0)
        if mode[0] == 'dragon':
            return None, None, x, y, t, None, None
        #
        img = torch.cat(img_, dim=0)
        one_hot = torch.cat(one_hot_, dim=0)
        if mode[0] == 'cls' or mode[0] == 'patch_cls':
            return img, None, x, y, t, one_hot, None
        #
        mask = torch.cat(mask_, dim=0)
        one_hot_tis = torch.cat(one_hot_tis_, dim=0)
        if mode[0] == 'tiscls' or mode[0] == 'total_cls':
            return img, mask, x, y, t, one_hot, one_hot_tis
        raise ValueError

    def padding_to_fixedsize(self, input, fixed_size, color=None):
        assert len(input.shape) == 3, f'the number of dimensions should be 3...'
        #
        if input.shape[0] == 3:
            pad_out = np.ones([input.shape[0], int(fixed_size), int(fixed_size)])
            pad_out *= color
        else:
            pad_out = np.zeros([input.shape[0], int(fixed_size), int(fixed_size)])
        #
        row_start_index = int(pad_out.shape[-2] / 2) - int(input.shape[-2] / 2)
        row_end_index = row_start_index + input.shape[-2]
        col_start_index = int(pad_out.shape[-1] / 2) - int(input.shape[-1] / 2)
        col_end_index = col_start_index + input.shape[-1]
        #
        if len(input.shape) == 1:
            pad_out[row_start_index:row_end_index, col_start_index:col_end_index] = input
        else:
            pad_out[:, row_start_index:row_end_index, col_start_index:col_end_index] = input
        #
        return pad_out


if __name__ == '__main__':
    from utils.config import Config

    #
    conf_file = '../config/lesion_dragon.py'
    cfg = Config.fromfile(conf_file)
    #
    dataset = lesionDataset(cfg, 'classify', 'train')
    #
    data = dataset.__getitem__(0)
    #
    from torch.utils.data.dataloader import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        num_workers=0,
        collate_fn=lesionDataset.collate_fn
    )

    for data in enumerate(loader):
        print(data)
