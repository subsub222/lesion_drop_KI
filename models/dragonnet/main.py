#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/kochbj/Deep-Learning-for-Causal-Inference/blob/main/Tutorial_2_Causal_Inference_Metrics_and_Hyperparameter_Optimization.ipynb
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataset import Lesion_data
from dragonnet import dragonnet
from loss import TarReg_Loss
import json
# tansorboard using
from torch.utils.tensorboard import SummaryWriter
# from utils import increment_name
from events import write_tbloss, write_tbacc, write_tbloss_val

# Seed
import random


def validation(model, val_loader, device, epoch, tblogger, tarreg_loss):
    model.eval()
    total_pred_yt = []
    total_yt = []
    total_validation_loss = []
    for batch_i, data in enumerate(val_loader):
        x = data['x'].to(device)
        yt = torch.cat([data['y'], data['t']], dim=1).to(device)
        preds = model(x)
        validation_loss = tarreg_loss._calc(yt, preds)

        total_yt.append(yt)
        total_pred_yt.append(preds)
        total_validation_loss.append(validation_loss)

    total_true_yt = torch.cat(total_yt, dim=0)
    total_pred_yt = torch.cat(total_pred_yt, dim=0)
    total_validation_loss = sum(total_validation_loss) / len(total_validation_loss)

    acc, pred_t, true_t = tarreg_loss.treatment_acc(total_true_yt, total_pred_yt)
    # write_tbacc(tblogger, acc, epoch, 'val')
    return acc, total_validation_loss


if __name__ == '__main__':
    # === configuration ===
    epoch = 500
    gpu_id = 0
    learninglate = 1e-5
    batch_size = 100

    # === sec, third dime setting ===
    hypo_dime_set = [700]
    sec_dime_set = [500]
    thir_dime_set = [500]
    results_val_acc = torch.zeros(len(sec_dime_set), len(thir_dime_set))
    results_val_total_acc = torch.zeros(len(sec_dime_set), len(thir_dime_set))
    results_train_acc = torch.zeros(len(sec_dime_set), len(thir_dime_set))
    results_train_total_acc = torch.zeros(len(sec_dime_set), len(thir_dime_set))

    # ===scheduler===
    sche_name = 'steplr'  # [steplr | plateau]
    opti_name = 'adam'  # [sgd | adam]

    # ===Device===
    device = torch.device('cuda:{}'.format(gpu_id))
    # device = torch.device('cpu')

    # === Dataset preprocessing ===
    train_df = pd.read_csv('/storage/yskim/open (1)/train.csv')
    test_df = pd.read_csv('/storage/yskim/open (1)/test.csv')

    train_df['암의 장경'] = train_df['암의 장경'].fillna(train_df['암의 장경'].mean())
    test_df['암의 장경'] = test_df['암의 장경'].fillna(test_df['암의 장경'].mean())

    train_df = train_df.drop(
        columns=['ID', 'img_path', 'mask_path', 'DCIS_or_LCIS_type', '수술연월일', 'HER2_SISH', 'HER2_SISH_ratio',
                 'BRCA_mutation'])
    train_df = train_df.fillna(0)
    test_df = test_df.drop(
        columns=['ID', 'img_path', '수술연월일', 'DCIS_or_LCIS_type', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation'])
    test_df = test_df.fillna(0)


    def get_values(value):
        return value.values.reshape(-1, 1)


    numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent']
    ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

    for col in train_df.columns:
        if col in ignore_cols:
            continue
        # standardscaler()==mean:0, deviation:1
        if col in numeric_cols:
            scaler = StandardScaler()
            train_df[col] = scaler.fit_transform(get_values(train_df[col]))
            test_df[col] = scaler.transform(get_values(test_df[col]))
        else:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(get_values(train_df[col]).ravel())
            test_df[col] = le.transform(get_values(test_df[col]).ravel())

    # === Crossvalidation ===
    train_df, val_df, train_labels, val_labels = train_test_split(
        train_df,
        train_df['N_category'],
        test_size=0.2,
        random_state=41
    )

    # ===make dataset===
    train_dataset = Lesion_data(training_data=train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              # collate_fn=train_dataset.collate_fn
                              )
    val_dataset = Lesion_data(training_data=val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            # collate_fn=train_dataset.collate_fn
                            )

    # === Start ===
    results_val_dic = {}
    results_train_dic = {}

    save_dir = str('./logging')
    tblogger = SummaryWriter(save_dir)

    for hypo_i, hypo_dime in enumerate(hypo_dime_set):
        for ten in range(10):
            for sec_i, sec_dime in enumerate(sec_dime_set):
                for thir_i, thir_dime in enumerate(thir_dime_set):
                    # === Network ===
                    model = dragonnet(in_dim=17, rep_dim=200, hypo_dim=hypo_dime, sec_dim=sec_dime, thir_dim=thir_dime,
                                      reg_l2=.01).to(device)

                    # === Optimizer ===
                    if opti_name == 'sgd':
                        optimizer = torch.optim.SGD(model.parameters(), lr=learninglate, momentum=0.9,
                                                    weight_decay=5e-5)
                    elif opti_name == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=learninglate, weight_decay=5e-5)
                    else:
                        optimizer = None
                        NotImplementedError('Not implemented optimizer...')

                    # === Scheduler ===
                    if sche_name == 'plateau':
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5,
                                                                               threshold=1e-3)
                    elif sche_name == 'steplr':
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
                    else:
                        scheduler = None
                        # NotImplementedError('Not implemented scheduler...')

                    # save directory for tensorboard
                    # save_dir = str(increment_name('./logging/exp'))

                    # save_dir = str('./logging/exp_{}_{}_{}'.format(hypo_dime, sec_dime, thir_dime))
                    # try:
                    #     os.makedirs(save_dir)
                    # except Exception as e:
                    #     print(e)

                    # Loss
                    tarreg_loss = TarReg_Loss(alpha=3., beta=3.)
                    total_val_acc = []
                    total_train_acc = []
                    max_stepnum = len(train_loader)
                    model.train()
                    for epoch_i in range(epoch):
                        model.train()
                        total_loss = []
                        total_acc = []
                        total_reg_loss = []
                        total_stand_loss = []
                        for batch_i, data in enumerate(train_loader):
                            x = data['x'].to(device)
                            yt = torch.cat([data['y'], data['t']], dim=1).to(device)

                            preds = model(x)
                            loss = tarreg_loss._calc(yt, preds)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            #
                            acc, pred_t, true_t = tarreg_loss.treatment_acc(yt, preds)
                            regloss = tarreg_loss.regression_loss(yt, preds)
                            stand_loss = tarreg_loss.standard_loss(yt, preds)
                            total_reg_loss.append(regloss)
                            total_acc.append(acc)
                            total_loss.append(loss)
                            total_stand_loss.append(stand_loss)
                            print(f'epoch : {epoch_i} hypo : {hypo_dime} sec : {sec_dime}, thir : {thir_dime} ',
                                  end='\r')
                            # if (batch_i + 1) % 7 == 0:
                            #     # print('epoch: {}'.format(epoch_i))
                            #     # print('avg. loss: {}'.format(loss))
                            #     # print('avg. reg loss: {}'.format(regloss))
                            #     # print('avg. standard loss: {}'.format(stand_loss))
                            #     # print('avg. acc: {}'.format(acc))
                            #     # pred_t = torch.sigmoid(pred_t)
                            #     # pred_t[pred_t > 0.5] = 1
                            #     # pred_t[pred_t <= 0.5] = 0
                            #     # print('t_pred: ', pred_t.detach().reshape(1, -1).cpu().numpy())
                            #     # print('t_true: ', true_t.detach().reshape(1, -1).cpu().numpy())
                            #     # print('--------------------------------------------------------------------')
                            #     write_tbloss(tblogger,
                            #                  losses=[(sum(total_loss)/len(total_loss)).detach().cpu().numpy(),
                            #                          (sum(total_reg_loss) / len(total_reg_loss)).detach().cpu().numpy(),
                            #                          (sum(total_stand_loss) / len(total_stand_loss)).detach().cpu().numpy()],
                            #                  step=(epoch_i * max_stepnum) + batch_i)
                        # Logging accuracy
                        # write_tbacc(tblogger, sum(total_acc) / len(total_acc), epoch_i, 'train')
                        val_acc, total_val_loss = validation(model, val_loader, device, epoch_i, tblogger, tarreg_loss)
                        # write_tbloss_val(tblogger, losses=[total_val_loss], step=epoch_i)
                        total_val_acc.append(val_acc)
                        total_train_acc.append(sum(total_acc) / len(total_acc))
                        # Update scheduler
                        if sche_name == 'plateau':
                            scheduler.step(sum(total_loss) / len(total_loss))
                        else:
                            scheduler.step()

                    results_val_acc[sec_i, thir_i] = sum(total_val_acc) / len(total_val_acc)
                    results_train_acc[sec_i, thir_i] = sum(total_train_acc) / len(total_train_acc)
            results_val_total_acc += results_val_acc
            results_train_total_acc += results_train_acc
        results_val_total_acc = results_val_total_acc / 10
        results_train_total_acc = results_train_total_acc / 10
        print('\n')

        results_val_dic[hypo_dime_set[hypo_i]] = results_val_total_acc.tolist()
        results_train_dic[hypo_dime_set[hypo_i]] = results_train_total_acc.tolist()
    print(results_val_dic)
    print(results_train_dic)
    # with open('./save_dict/results_val_dicsub.json', 'w') as f:
    #     json.dump(results_val_dic, f, indent=4)
    # with open('./save_dict/results_train_dicsub.json', 'w') as f:
    #     json.dump(results_train_dic, f, indent=4)

    print('a')
    # Evaluation
    # model.eval()
    # data_mu_0 = []
    # data_mu_1 = []
    # y0_pred_all = []
    # y1_pred_all = []
    # for batch_i, data in enumerate(train_loader):
    #     x = data['x'].to(device)
    #     preds = model(x)
    #     #
    #     y_preds = tarreg_loss.split_pred(preds)
    #     y0_pred, y1_pred = y_preds['y0_pred'], y_preds['y1_pred']
    #     #
    #     data_mu_0.append(data['mu_0'])
    #     data_mu_1.append(data['mu_1'])
    #     y0_pred_all.append(y0_pred.detach().cpu())
    #     y1_pred_all.append(y1_pred.detach().cpu())
    #
    # y_scaler = train_dataset.scaler
    # mu_0 = np.concatenate(data_mu_0, axis=0)
    # mu_1 = np.concatenate(data_mu_1, axis=0)
    # y0_pred = np.concatenate(y0_pred_all, axis=0)
    # y1_pred = np.concatenate(y1_pred_all, axis=0)
    #
    # plot_cates(np.expand_dims(y0_pred, axis=1),
    #            np.expand_dims(y1_pred, axis=1),
    #            y_scaler,
    #            np.expand_dims(mu_1, axis=1), np.expand_dims(mu_0, axis=1),
    #            rep_dime,
    #            hypo_dime,
    #            sec_dime,thir_dime,learninglate)
    # print('./cateplot/plot_epoch{}_repdim_{}_hdim{}_sdim{}_thirdim{}_lr{}_suffle0_sechY2.png'.format(epoch, rep_dime,hypo_dime, sec_dime, thir_dime,learninglate))
    # plt.plot(lossplot)
    # plt.savefig('./total_lossplot/loss{}.png'.format(epoch))
    # plt.close()
    # plt.plot(accplot)
    # plt.savefig('./accplot/acc{}.png'.format(epoch))
    # plt.close()
    # plt.plot(reg_lossplot)
    # plt.savefig('./regplot/regloss{}.png'.format(epoch))
    # plt.close()

    # Model save
    # state = {'net': model.state_dict(), 'opt': optimizer.state_dict}
    #
    # torch.save(state, './Check_point/plot_epoch{}_repdim_{}_hdim{}_sdim{}_thirdim{}_lr{}.pth')
