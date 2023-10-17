import os
import random
import json
import numpy as np

from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def split_train_val(path, train_val_ratio):
    numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent']
    ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

    df = pd.read_csv(path)
    df['암의 장경'] = df['암의 장경'].fillna(df['암의 장경'].mean())
    # 'ID', 'img_path', 'mask_path',
    df = df.fillna(0)
    # df = df.drop(columns=['DCIS_or_LCIS_type', '수술연월일', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation'])
    df = df.drop(columns=['수술연월일'])

    #
    def get_values(value):
        return value.values.reshape(-1, 1)

    for col in df.columns:
        if col in ignore_cols:
            continue
        elif col in numeric_cols:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(get_values(df[col]))
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(get_values(df[col]).ravel())
    #
    y_data = df.drop(
        # columns=['나이', '진단명', '암의 위치', '암의 개수', 'ER',
        #          'ER_Allred_score', 'PR', 'PR_Allred_score', 'HER2',
        #          'HER2_IHC', 'N_category', 'NG', 'HG', 'HG_score_1',
        #          'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_여부',
        #          'T_category']
        columns=['나이', '진단명', '암의 위치', '암의 개수', 'ER',
                 'ER_Allred_score', 'PR', 'PR_Allred_score', 'HER2',
                 'HER2_IHC', 'N_category', 'NG', 'HG', 'HG_score_1',
                 'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_여부',
                 'T_category', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation', 'DCIS_or_LCIS_type']
    )
    y_keys = list(y_data.keys())
    y_data = np.array(y_data)
    #
    x_data = df.drop(
        columns=['암의 장경']
        # columns=['암의 장경', 'KI-67_LI_percent', 'N_category']
    )
    x_keys = list(x_data.keys())
    x_data = np.array(x_data)
    #
    t_data = df.drop(columns=['나이', '진단명', '암의 위치', '암의 개수', '암의 장경', 'NG', 'HG', 'HG_score_1',
                              'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_여부', 'T_category', 'ER',
                              'ER_Allred_score', 'PR', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2',
                              'HER2_IHC', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation', 'DCIS_or_LCIS_type'])
    t_keys = list(t_data.keys())
    t_data = np.array(t_data)
    # ===== split =====
    mask_idx = list(np.where(np.array(x_data[:, 2]) != '-')[0])
    no_mask_idx = list(np.where(np.array(x_data[:, 2]) == '-')[0])
    #
    x_data_mask = x_data[np.array(mask_idx)]
    x_data_wo_mask = x_data[np.array(no_mask_idx)]
    y_data_mask = y_data[np.array(mask_idx)]
    y_data_wo_mask = y_data[np.array(no_mask_idx)]
    t_data_mask = t_data[np.array(mask_idx)]
    t_data_wo_mask = t_data[np.array(no_mask_idx)]
    #
    metas_idxes_mask = np.where(t_data_mask[:, 3] == 1)[0]
    no_metas_idxes_mask = np.where(t_data_mask[:, 3] == 0)[0]
    metas_idxes_wo_mask = np.where(t_data_wo_mask[:, 3] == 1)[0]
    no_metas_idxes_wo_mask = np.where(t_data_wo_mask[:, 3] == 0)[0]
    #
    num_train_metas_w_mask = int(train_val_ratio * len(metas_idxes_mask))
    num_train_no_metas_w_mask = int(train_val_ratio * len(no_metas_idxes_mask))
    num_train_metas_wo_mask = int(train_val_ratio * len(metas_idxes_wo_mask))
    num_train_no_metas_wo_mask = int(train_val_ratio * len(no_metas_idxes_wo_mask))
    #
    assert (metas_idxes_mask.shape[0] + no_metas_idxes_mask.shape[0] +
            metas_idxes_wo_mask.shape[0] + no_metas_idxes_wo_mask.shape[0]) == len(df)
    metas_idxes_mask = list(metas_idxes_mask)
    no_metas_idxes_mask = list(no_metas_idxes_mask)
    metas_idxes_wo_mask = list(metas_idxes_wo_mask)
    no_metas_idxes_wo_mask = list(no_metas_idxes_wo_mask)
    train_metas_w_mask_idxes = random.sample(metas_idxes_mask, num_train_metas_w_mask)
    train_no_metas_w_mask_idxes = random.sample(no_metas_idxes_mask, num_train_no_metas_w_mask)
    train_metas_wo_mask_idxes = random.sample(metas_idxes_wo_mask, num_train_metas_wo_mask)
    train_no_metas_wo_mask_idxes = random.sample(no_metas_idxes_wo_mask, num_train_no_metas_wo_mask)
    for i in train_metas_w_mask_idxes:
        metas_idxes_mask.remove(i)
    test_metas_w_mask_idxes = metas_idxes_mask
    del metas_idxes_mask
    for i in train_no_metas_w_mask_idxes:
        no_metas_idxes_mask.remove(i)
    test_no_metas_w_mask_idxes = no_metas_idxes_mask
    del no_metas_idxes_mask
    for i in train_metas_wo_mask_idxes:
        metas_idxes_wo_mask.remove(i)
    test_metas_wo_mask_idxes = metas_idxes_wo_mask
    del metas_idxes_wo_mask
    for i in train_no_metas_wo_mask_idxes:
        no_metas_idxes_wo_mask.remove(i)
    test_no_metas_wo_mask_idxes = no_metas_idxes_wo_mask
    del no_metas_idxes_wo_mask
    assert (len(train_metas_w_mask_idxes) + len(test_metas_w_mask_idxes)
            + len(train_no_metas_w_mask_idxes) + len(test_no_metas_w_mask_idxes)
            + len(train_metas_wo_mask_idxes) + len(test_metas_wo_mask_idxes)
            + len(train_no_metas_wo_mask_idxes) + len(test_no_metas_wo_mask_idxes)
            == len(x_data)), f'splitting procedure failed...'
    #
    train_x_wo_mask = (
            list(x_data_wo_mask[train_metas_wo_mask_idxes]) + list(x_data_wo_mask[train_no_metas_wo_mask_idxes]))
    train_y_wo_mask = (
            list(y_data_wo_mask[train_metas_wo_mask_idxes]) + list(y_data_wo_mask[train_no_metas_wo_mask_idxes]))
    train_t_wo_mask = (
            list(t_data_wo_mask[train_metas_wo_mask_idxes]) + list(t_data_wo_mask[train_no_metas_wo_mask_idxes]))
    test_x_wo_mask = (
            list(x_data_wo_mask[test_metas_wo_mask_idxes]) + list(x_data_wo_mask[test_no_metas_wo_mask_idxes]))
    test_y_wo_mask = (
            list(y_data_wo_mask[test_metas_wo_mask_idxes]) + list(y_data_wo_mask[test_no_metas_wo_mask_idxes]))
    test_t_wo_mask = (
            list(t_data_wo_mask[test_metas_wo_mask_idxes]) + list(t_data_wo_mask[test_no_metas_wo_mask_idxes]))
    train_x_mask = list(x_data_mask[train_metas_w_mask_idxes]) + list(x_data_mask[train_no_metas_w_mask_idxes])
    test_x_mask = list(x_data_mask[test_metas_w_mask_idxes]) + list(x_data_mask[test_no_metas_w_mask_idxes])
    train_y_mask = list(y_data_mask[train_metas_w_mask_idxes]) + list(y_data_mask[train_no_metas_w_mask_idxes])
    test_y_mask = list(y_data_mask[test_metas_w_mask_idxes]) + list(y_data_mask[test_no_metas_w_mask_idxes])
    train_t_mask = list(t_data_mask[train_metas_w_mask_idxes]) + list(t_data_mask[train_no_metas_w_mask_idxes])
    test_t_mask = list(t_data_mask[test_metas_w_mask_idxes]) + list(t_data_mask[test_no_metas_w_mask_idxes])
    assert len(train_x_wo_mask) + len(test_x_wo_mask) + len(train_x_mask) + len(test_x_mask) == len(
        df), f'Invalid splitting...'
    #
    base_path = '/'.join(path.split('/')[:-1])
    train_data = dict()
    train_x_wo_mask = np.array(train_x_wo_mask)
    for i in tqdm(range(train_x_wo_mask.shape[0]), total=len(train_x_wo_mask), leave=True, ascii=True,
                  postfix='train data'):
        x = list(train_x_wo_mask[i])
        y = list(train_y_wo_mask[i])
        t = list(train_t_wo_mask[i])
        id = x[0]
        img_path = os.path.join(base_path, '/'.join(x[1].split('/')[-2:]))
        mask_path = os.path.join(base_path, '/'.join(x[1].split('/')[-2:])) if x[2].split('/')[-2:][
                                                                                   0] != '-' else '-'
        assert os.path.exists(img_path), f'There is no image...'
        if mask_path != '-':
            assert os.path.exists(mask_path), f'There is no mask image...'
        #
        train_data[img_path] = {'img_path': img_path, 'mask_path': mask_path, 'id': id,
                                'x': x[3:], 'y': y[3:], 't': t[3:],
                                'x_keys': x_keys[3:], 'y_keys': y_keys[3:], 't_keys': t_keys[3:]}
    test_data = dict()
    test_x_wo_mask = np.array(test_x_wo_mask)
    for i in tqdm(range(test_x_wo_mask.shape[0]), total=len(test_x_wo_mask), leave=True, ascii=True,
                  postfix='test data'):
        x = list(test_x_wo_mask[i])
        y = list(test_y_wo_mask[i])
        t = list(test_t_wo_mask[i])
        id = x[0]
        img_path = os.path.join(base_path, '/'.join(x[1].split('/')[-2:]))
        mask_path = os.path.join(base_path, '/'.join(x[1].split('/')[-2:])) if x[2].split('/')[-2:][
                                                                                   0] != '-' else '-'
        assert os.path.exists(img_path), f'There is no image...'
        if mask_path != '-':
            assert os.path.exists(mask_path), f'There is no mask image...'
        #
        test_data[img_path] = {'img_path': img_path, 'mask_path': mask_path, 'id': id,
                               'x': x[3:], 'y': y[3:], 't': t[3:],
                               'x_keys': x_keys[3:], 'y_keys': y_keys[3:], 't_keys': t_keys[3:]}

    train_data_mask = dict()
    train_x_mask = np.array(train_x_mask)
    for i in tqdm(range(train_x_mask.shape[0]), total=len(train_x_mask), leave=True, ascii=True,
                  postfix='train data with mask'):
        x = list(train_x_mask[i])
        y = list(train_y_mask[i])
        t = list(train_t_mask[i])
        id = x[0]
        img_path = os.path.join(base_path, 'ori_train_image_gray', x[2].split('/')[-1:][0])
        mask_path = os.path.join(base_path, 'gt_train_image', x[2].split('/')[-1:][0]) if x[2].split('/')[-2:][
                                                                                              0] != '-' else '-'
        assert os.path.exists(img_path), f'There is no image...'
        if mask_path != '-':
            assert os.path.exists(mask_path), f'There is no mask image...'
        #
        train_data_mask[img_path] = {'img_path': img_path, 'mask_path': mask_path, 'id': id,
                                     'x': x[3:], 'y': y[3:], 't': t[3:],
                                     'x_keys': x_keys[3:], 'y_keys': y_keys[3:], 't_keys': t_keys[3:]}
        train_data[img_path] = {'img_path': img_path, 'mask_path': mask_path, 'id': id,
                                'x': x[3:], 'y': y[3:], 't': t[3:],
                                'x_keys': x_keys[3:], 'y_keys': y_keys[3:], 't_keys': t_keys[3:]}
    test_data_mask = dict()
    test_x_mask = np.array(test_x_mask)
    for i in tqdm(range(test_x_mask.shape[0]), total=len(test_x_mask), leave=True, ascii=True,
                  postfix='test data with mask'):
        x = list(test_x_mask[i])
        y = list(test_y_mask[i])
        t = list(test_t_mask[i])
        id = x[0]
        img_path = os.path.join(base_path, 'ori_train_image_gray', x[2].split('/')[-1:][0])
        mask_path = os.path.join(base_path, 'gt_train_image', x[2].split('/')[-1:][0]) if x[2].split('/')[-2:][
                                                                                              0] != '-' else '-'
        assert os.path.exists(img_path), f'There is no image...'
        if mask_path != '-':
            assert os.path.exists(mask_path), f'There is no mask image...'
        #
        test_data_mask[img_path] = {'img_path': img_path, 'mask_path': mask_path, 'id': id,
                                    'x': x[3:], 'y': y[3:], 't': t[3:],
                                    'x_keys': x_keys[3:], 'y_keys': y_keys[3:], 't_keys': t_keys[3:]}
        test_data[img_path] = {'img_path': img_path, 'mask_path': mask_path, 'id': id,
                               'x': x[3:], 'y': y[3:], 't': t[3:],
                               'x_keys': x_keys[3:], 'y_keys': y_keys[3:], 't_keys': t_keys[3:]}

    total_data = dict()
    total_data['train'] = train_data
    total_data['test'] = test_data
    total_data['train_mask'] = train_data_mask
    total_data['test_mask'] = test_data_mask

    with open(os.path.join(base_path, 'splitted_data_gray.json'), 'w') as f:
        json.dump(total_data, f)


if __name__ == '__main__':
    from utils.config import Config

    #
    conf_file = '../config/lesion_dragon.py'
    cfg = Config.fromfile(conf_file)
    # split
    split_train_val(os.path.join(cfg.dataset.base_path, 'train.csv'), train_val_ratio=0.8)
