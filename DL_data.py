from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import os, time, pickle
import pandas as pd
# from model_ult import regression_method
from Features2img import get_map_features
import cls_ult
import joblib

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



def pre_data(map_data, map_mask, need_filter=False):
    from scipy.interpolate import griddata
    import scipy.signal as signal
    points = np.nonzero(map_mask)
    values = map_data[map_mask.astype('bool')]  # 已知散点的值
    # print(len(points), len(values))
    xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    # print(xi, yi)
    znew = griddata(points, values, (xi, yi), method='nearest')  # 进行插值
    if need_filter :
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波
    return znew

def get_draw_map(features, w, h):
    RSRP = features['RSRP'].values
    PL = features['RS Power'].values - RSRP
    # 7.用户海拔m
    Husr = features['Altitude'].values
    # 8.用户建筑物高度m
    Hm = features['Building Height'].values
    CI = features['Clutter Index'].values
    x_ = np.array(features['x_'].values).astype(np.int) + w/2
    y_ = np.array(features['y_'].values).astype(np.int)

    # 3.垂直下倾角
    betaV = features['Electrical Downtilt'].values + features['Mechanical Downtilt'].values
    # 6.基站塔顶高度m
    Hb = features['Height'].values + features['Cell Building Height'].values + features['Cell Altitude'].values
    # 10.水平距离m
    x = features['X'].values - features['Cell X'].values
    y = features['Y'].values - features['Cell Y'].values
    L = np.sqrt(np.square(x) + np.square(y))
    # 16.信号线与用户栅格建筑顶的相对高度m
    deltaHv = Hb - Husr - Hm - (L * np.tan(np.deg2rad(betaV)))

    map_att = np.zeros([w, h])
    map_mask = np.zeros([w, h])
    map_rsrp = np.full([w, h], np.nan)
    map_building = np.zeros([w, h])
    map_ci = np.zeros([w, h])
    map_deltaHv = np.zeros([w, h])

    # col = []

    for i in range(len(x_)):
        if x_[i] < w:
            if y_[i] < h:

                xx = int(x_[i])
                yy = int(y_[i])

                map_mask[xx, yy] = 1
                map_rsrp[xx, yy] = RSRP[i]
                map_att[xx, yy] = Husr[i]
                # map_deltaHv[xx, yy] = deltaHv[i]
                map_building[xx, yy] = Hm[i]
                map_ci[xx, yy] = CI[i]
                map_deltaHv[xx, yy] = deltaHv[i]

    return map_mask, map_att, map_building, map_deltaHv, map_ci, map_rsrp


from matplotlib import cm
def draw_inputs_2D(data, title, map_rsrp, ML=False):
    import matplotlib as mpl

    fig,ax = plt.subplots(1, 1, figsize=(8,8))

    surf = ax.imshow(data,
                     # norm=vnorm,
                     alpha=0.26,
                     # cmap=cm.coolwarm
                     )
    data = data+0.01
    col = np.nonzero(data)
    # map_rsrp = map_rsrp[map_mask.astype('bool')]

    # if not ML:
    #     map_rsrp = map_rsrp[map_mask.astype('bool')]

    # c_list = (map_rsrp - map_rsrp.min()) / (map_rsrp.max() - map_rsrp.min())
    a = ax.scatter(col[1], col[0],
               c=map_rsrp,
               cmap='coolwarm',
               s=10,
               alpha=1.0,
               label='RSRP')


    ax.scatter(0, 32,
               color='red',
               s=100,
               marker='s',
               alpha=1.0,
               label='BS')

    ax.set_title(title)

    norm = mpl.colors.Normalize(vmin=-130, vmax=-60)
    fig.colorbar(a,norm=norm,fraction=0.048, pad=0.02)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()


def get_ML_pred(features):
    map_mask, map_att, map_building, map_deltaHv, map_ci, map_rsrp = get_draw_map(features, 64, 64)
    map_building = pre_data(map_building, map_mask)
    map_att = pre_data(map_att, map_mask, need_filter=False)
    map_data = map_building + map_att

    map_rsrp = pre_data(map_rsrp, map_mask, need_filter=False)
    # print(np.shape(map_data))

    att_min = features['Altitude'].min()
    f = features['Frequency Band'].values[0]

    a = features['Azimuth'].values[0]

    h_bs = features['Cell Altitude'].values[0] + \
           features['Cell Building Height'].values[0] + \
           features['Height'].values[0] - att_min

    betaV = features['Electrical Downtilt'].values[0] + features['Mechanical Downtilt'].values[0]

    RSP = features['RS Power'].values[0]

    h_ue = map_att - att_min + 1.0
    Hm = map_building

    xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    # print(xi-32, yi)
    d = np.sqrt(np.square(5 * (xi - 32)) + np.square(5 * (yi)))
    # print(np.shape(d))

    ones = np.ones(64 * 64)

    pfs = np.array([f * ones, np.reshape(d, [64 * 64]), np.reshape(Hm, [64 * 64]),
                    np.reshape(h_ue, [64 * 64]), betaV * ones, a * ones, RSP * ones])
    pfs = pfs.swapaxes(0, 1)
    print(np.shape(pfs))

    # f, d, Hm, h_ue, deltaH, deltaHv, betaV, RSP, a, RSRP \
    #     = get_ML_features(features)
    #
    # pfs = np.array([f, d, Hm, h_ue, betaV, a, RSP])

    y_pred = model.predict(pfs)
    y_pred = np.reshape(y_pred, [64, 64])

    return y_pred, map_data, map_rsrp, map_mask

def maxmin_norm(data, mask, is_MASK=True):
    if is_MASK:
        data = (data-data[mask.astype('bool')].min()) \
                     / (data[mask.astype('bool')].max()
                        - data[mask.astype('bool')].min())
        data = data * mask
    else:
        data = (data - data.min()) / (data.max() - data.min())
    return data


model = joblib.load("./model/ML/rfr_sc.pkl")

from gd_rsrp import Discriminator, Generator
generator = Generator()
model_dir = "./model/GAN/GAN"
generator.load_weights(model_dir)

data_path = "./dataset/csv_set_h50_64/testset/"
filenames = cls_ult.get_filename(data_path)
train_dataset, test_dataset = [], []
cnt = 0
for name, context in filenames:
    cnt += 1
    # print(name,context)
    features = pd.read_csv(context)

    train_features = features.sample(frac=0.8)
    y_pred, map_data, map_rsrp, map_mask = get_ML_pred(train_features)
    train_data = np.array([y_pred, map_data, map_rsrp, map_mask])


    y_pred, map_data, map_rsrp, map_mask = get_ML_pred(features)
    test_data = np.array([y_pred, map_data, map_rsrp, map_mask])

    # draw_inputs_2D(map_data, '', y_pred)
    # draw_inputs_2D(map_data, '', map_rsrp)

    # print("y_pred:",np.isnan(y_pred).any())
    # print("map_data:", np.isnan(map_data).any())
    # print("map_rsrp:", np.isnan(map_rsrp).any())
    # print("map_mask:", np.isnan(map_mask).any())

    # print(map_rsrp*map_mask)
    y_pred = np.reshape(y_pred, [1, 64, 64, 1])
    map_data = np.reshape(map_data, [1, 64, 64, 1])

    ground_true = np.reshape(map_rsrp, [1, 64, 64, 1])
    mask = np.reshape(map_mask, [1, 64, 64, 1])

    ML_pred = maxmin_norm(y_pred, mask, is_MASK=False)
    map_data = maxmin_norm(map_data, map_mask, is_MASK=False)

    prediction = generator(ML_pred, training=False)
    prediction = prediction * (ground_true[mask.astype('bool')].max()
                               - ground_true[mask.astype('bool')].min()) \
                 + ground_true[mask.astype('bool')].min()

    map_data = np.reshape(map_data, [64, 64, 1])
    prediction = np.reshape(prediction, [64, 64, 1])

    draw_inputs_2D(map_data, '', prediction)

    train_dataset.append(train_data)
    test_dataset.append(test_data)

    if cnt == 1:
        break
# print(dataset)
# print("dataset NAN?:", np.isnan(train_dataset).any())
#
# np.save('./dataset/train_dataset.npy',train_dataset)
# np.save('./dataset/test_dataset.npy',test_dataset)

