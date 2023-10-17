import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


# origin load_IHDP_data

def load_Lesion_data(training_data, testing_data, i=7):
    with open(training_data, 'rb') as trf, open(testing_data, 'rb') as tef:
        train_data = np.load(trf)
        test_data = np.load(tef)
        y = np.concatenate((train_data['yf'][:, i], test_data['yf'][:, i])).astype(
            'float32')  # most GPUs only compute 32-bit floats
        t = np.concatenate((train_data['t'][:, i], test_data['t'][:, i])).astype('float32')
        x = np.concatenate((train_data['x'][:, :, i], test_data['x'][:, :, i]), axis=0).astype('float32')
        mu_0 = np.concatenate((train_data['mu0'][:, i], test_data['mu0'][:, i])).astype('float32')
        mu_1 = np.concatenate((train_data['mu1'][:, i], test_data['mu1'][:, i])).astype('float32')

        data = {'x': x, 't': t, 'y': y, 'mu_0': mu_0, 'mu_1': mu_1}
        data['t'] = data['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with an additional dimension
        data['y'] = data['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data['y_scaler'] = StandardScaler().fit(data['y'])
        data['ys'] = data['y_scaler'].transform(data['y'])
    return data


class Lesion_data(Dataset):
    def __init__(self, training_data):
        super().__init__()

        self.train_data = training_data
        self.ytrain_data = np.array(self.train_data.drop(columns=['나이', '진단명', '암의 위치', '암의 개수', 'ER',
                                                                  'ER_Allred_score', 'PR', 'PR_Allred_score', 'HER2',
                                                                  'HER2_IHC', 'N_category', 'NG', 'HG', 'HG_score_1',
                                                                  'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_여부',
                                                                  'T_category']))
        self.xtrain_data = np.array(self.train_data.drop(columns=['암의 장경', 'KI-67_LI_percent', 'N_category']))
        self.ttrain_data = np.array(self.train_data.drop(columns=['나이', '진단명', '암의 위치', '암의 개수', '암의 장경', 'NG', 'HG', 'HG_score_1',
                                          'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_여부', 'T_category', 'ER',
                                          'ER_Allred_score', 'PR', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2',
                                          'HER2_IHC']))
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        y = (self.ytrain_data[index]).astype('float32')  # most GPUs only compute 32-bit floats
        t = (self.ttrain_data[index]).astype('float32')
        x = (self.xtrain_data[index]).astype('float32')
        data = {'x': torch.tensor(x), 't': torch.tensor(t), 'y': torch.tensor(y)}
        # rescaling y between 0 and 1 often makes training of DL regressors easier

        return data

    def collate_fn(self, batch):
        assert len(batch) == 1
        batch = batch[0]
        return batch


# origin getitem(concatenate train and test dataset)
# X_tr, T_tr, YF_tr, YCF_tr, mu_0_tr, mu_1_tr are covariates, treatment, factual outcome, counterfactual outcome, and noiseless potential outcomes respectively;'''
'''
    def __getitem__(self, index):
        print(index)
        y = np.concatenate((self.train_data['yf'][:, index], self.test_data['yf'][:, index])).astype(
            'float32')  # most GPUs only compute 32-bit floats
        t = np.concatenate((self.train_data['t'][:, index], self.test_data['t'][:, index])).astype('float32')
        x = np.concatenate((self.train_data['x'][:, :, index], self.test_data['x'][:, :, index]), axis=0).astype(
            'float32')
        mu_0 = np.concatenate((self.train_data['mu0'][:, index], self.test_data['mu0'][:, index])).astype('float32')
        mu_1 = np.concatenate((self.train_data['mu1'][:, index], self.test_data['mu1'][:, index])).astype('float32')

        data = {'x': torch.tensor(x), 't': torch.tensor(t), 'y': torch.tensor(y),
                'mu_0': torch.tensor(mu_0), 'mu_1': torch.tensor(mu_1)}
        # we're just padding one dimensional vectors with an additional dimension
        data['t'] = data['t'].reshape(-1, 1)
        data['y'] = data['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data['y_scaler'] = StandardScaler().fit(data['y'])
        data['ys'] = torch.tensor(data['y_scaler'].transform(data['y']))

        # data['y_scaler'] = StandardScaler().fit(data['y'])
        # data['ys'] = data['y_scaler'].transform(data['y'])
        return data
'''

if __name__ == '__main__':
    data = load_IHDP_data(training_data='/storage/yskim/casual/ihdp_npci_1-100.train.npz',
                          testing_data='/storage/yskim/casual/ihdp_npci_1-100.test.npz')
    # class
    data_object = IHDP_data(training_data='/storage/yskim/casual/ihdp_npci_1-100.train.npz',
                            testing_data='/storage/yskim/casual/ihdp_npci_1-100.test.npz')
    # concatenate t so we can use it as input
    xt = np.concatenate([data['x'], data['t']], 1)

    #
    data_loader = iter(data_object)
    data = data_loader.__next__()
    xt = np.concatenate([data['x'], data['t']], 1)
