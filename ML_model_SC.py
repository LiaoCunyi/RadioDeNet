from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import os, time, pickle
import pandas as pd
from model_ult import regression_method
from Features2img import get_map_features
import cls_ult

def get_features(pb_data):
    x = pb_data['X'] - pb_data['Cell X']
    y = pb_data['Y'] - pb_data['Cell Y']
    # d = np.sqrt(np.square(x) + np.square(y)).values / 1000

    # 1.小区标识-
    CI = pb_data['Cell Index']

    # 2.中心频率MHz
    FB = pb_data['Frequency Band']
    # FB = np.log10(pb_data['Frequency Band'])

    # 3.垂直下倾角
    betaV = pb_data['Electrical Downtilt'] + pb_data['Mechanical Downtilt']

    # 4.发射功率dBm
    RSP = pb_data['RS Power']

    # 5.基站地物索引-
    CCI = pb_data['Cell Clutter Index']

    # 6.基站塔顶高度m
    Hb = pb_data['Height'] + pb_data['Cell Building Height'] + pb_data['Cell Altitude']

    # 7.用户海拔m
    Husr = pb_data['Altitude']

    # 8.用户建筑物高度m
    Hm = pb_data['Building Height']

    # 9.有效高度m
    deltaH = Hb - Husr - Hm

    # 10.水平距离m
    L = np.sqrt(np.square(x) + np.square(y))

    # 11.链路距离m
    d = np.sqrt(np.square(L) + np.square(deltaH))


    RSRP = pb_data['RSRP'].values

    return 0

def get_ML_features(features):
    features = features.sample(frac=1).reset_index(drop=True)

    att_min = features['Altitude'].min()

    f = features['Frequency Band'].values

    a = features['Azimuth'].values

    h_bs = features['Cell Altitude'].values + \
           features['Cell Building Height'].values + \
           features['Height'].values - att_min
    h_ue = features['Altitude'].values \
           - att_min + 1.0

    x = features['X'] - features['Cell X']
    y = features['Y'] - features['Cell Y']
    d = np.sqrt(np.square(x) + np.square(y)).values

    # 3.垂直下倾角
    betaV = features['Electrical Downtilt'].values + features['Mechanical Downtilt'].values

    # 8.用户建筑物高度m
    Hm = features['Building Height'].values
    # 9.有效高度m
    deltaH = h_bs - h_ue - Hm
    # 16.信号线与用户栅格建筑顶的相对高度m
    deltaHv = deltaH - (d * np.tan(np.deg2rad(betaV)))

    # print("f:", f[0], "d:", d[0], "h_bs:", h_bs[0], "h_ue:", h_ue[0])
    rsrp = features['RSRP'].values
    RSP = features['RS Power'].values


    log_d = np.log10(d)
    log_h_bs = np.log10(h_bs)
    # log_h_ue = np.log10(h_ue)
    return f, d, Hm, h_ue, deltaH, deltaHv, betaV, RSP, a, np.array([rsrp])
    # return f, log_d, log_h_bs, h_ue, deltaH, betaV, np.array([PL])


def get_dataset(data_path, is_training = True):
    filenames = cls_ult.get_filename(data_path)
    train_data = np.zeros([7,1])
    train_label = np.zeros([1,1])

    test_data = np.zeros([7, 1])
    test_label = np.zeros([1, 1])
    cnt = 0
    for name,context in filenames:
        cnt+=1
        # print(name,context)
        features = pd.read_csv(context)
        f, d, Hm, h_ue, deltaH, deltaHv, betaV, RSP, a, RSRP \
            = get_ML_features(features)


        pfs = np.array([f, d, Hm, h_ue, betaV, a, RSP])
        # print("pfs:",np.shape(pfs))

        if cnt <= 0.8 * len(filenames):
            # PL = (PL - 61.2) / 90.9
            train_data = np.concatenate([train_data, pfs], axis=1)
            train_label = np.concatenate([train_label, RSRP], axis=1)
        # else:
        test_data = np.concatenate([test_data, pfs], axis=1)
        test_label = np.concatenate([test_label, RSRP], axis=1)
        # map_deltaH, map_deltaHv, map_mask, map_rsrp = get_map_features(features, weight, height)
        # break
    return train_data.T[1:], train_label.T[1:], test_data.T[1:], test_label.T[1:]


print("Read dataset...")
# train_filenames = cls_ult.get_filename("./dataset/csv_set_h50_64/trainset/")
# test_filenames = cls_ult.get_filename("./dataset/csv_set_h50_64/testset/")
X_train, Y_train, X_test, Y_test = get_dataset("./dataset/csv_set_h50_64/all/",is_training = True)
print("X_train:",np.shape(X_train))
# X_test, Y_test = get_dataset("./dataset/csv_set_h50_64/testset/", is_training = False)

# dataset = np.load("./dataset/Set_hd50_82.npy")
# rsp = np.load("./dataset/rsp_Set_hd50_82.npy")

print("train_data shape:",np.shape(X_train),"train_label shape:",np.shape(Y_train))
print("test_data shape:",np.shape(X_test),"test_label shape:",np.shape(Y_test))

print(X_train[:10])


tailname = '_sc'
need_test = True
max_depth = 20
n_estimators = 20

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
regression_method("lr"+tailname, lr,
                  X_train, X_test, Y_train, Y_test, need_test=need_test)



from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(weights="uniform")
regression_method("knn"+tailname, knn,
                  X_train, X_test, Y_train, Y_test, need_test=need_test)

# from sklearn import tree
# dtr = tree.DecisionTreeRegressor(max_depth=max_depth)
# regression_method("dtr"+tailname, dtr,
#                   X_train, X_test, Y_train, Y_test, need_test=need_test)

from sklearn import ensemble
rfr = ensemble.RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
regression_method("rfr"+tailname, rfr,
                  X_train, X_test, Y_train, Y_test, need_test=need_test)
#
# from sklearn.tree import ExtraTreeRegressor
# etr = ExtraTreeRegressor(max_depth=max_depth,)
# regression_method("etr"+tailname, etr,
#                   X_train, X_test, Y_train, Y_test, need_test=need_test)
#
Adaboost = ensemble.AdaBoostRegressor(n_estimators=n_estimators)
regression_method("Adaboost"+tailname, Adaboost,
                  X_train, X_test, Y_train, Y_test, need_test=need_test)

# gbrt = ensemble.GradientBoostingRegressor(n_estimators=n_estimators)
# regression_method("gbrt"+tailname, gbrt,
#                   X_train, X_test, Y_train, Y_test, need_test=need_test)
#
#
# bagging = ensemble.BaggingRegressor()
# regression_method("bagging"+tailname, bagging,
#                   X_train, X_test, Y_train, Y_test, need_test=need_test)



# from sklearn.svm import SVR
# l_svr = SVR(kernel='linear')
# regression_method("svr_linear"+tailname, l_svr,
#                   X_train, X_test, Y_train, Y_test, need_test=need_test)

# r_svr = SVR(kernel="rbf")
# regression_method("svr_rbf"+tailname, r_svr,
#                   X_train, X_test, Y_train, Y_test, need_test=need_test)