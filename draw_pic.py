from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import cls_ult
weight, height = 64,64

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

from sklearn.impute import KNNImputer
def img_imputer(input_image, mask):
    input_image = np.array(np.reshape(input_image, [weight, height]))
    mask = np.reshape(mask, [weight, height])
    input_image[~mask.astype('bool')] = np.nan
    # input_image = np.nan_to_num(input_image)
    # print(np.shape(input_image), np.mean(input_image))
    # print("Before: ", np.shape(input_image))
    imputer = KNNImputer(n_neighbors=1)
    output_image = imputer.fit_transform(input_image)
    # print("After: ", np.shape(output_image))
    # output_image = np.pad(output_image,((0,weight-np.shape(output_image)[0]),(0,height-np.shape(output_image)[1])),'constant',constant_values=1)
    # print("After that: ", np.shape(output_image))
    # output_image = np.reshape(output_image,[batch_size,weight,height,input_image_channel])
    return output_image

def pre_data(map_data, map_mask, need_filter=True):
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

def draw_building_3D(map_data, map_mask,map_rsrp, features):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)

    col = np.nonzero(map_mask)
    map_rsrp = map_rsrp[map_mask.astype('bool')]



    # 设置柱子属性
    height = map_data.max()  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 1  # 柱子的长和宽

    # X = 32
    # Y = 0
    # Z = features['Height'].values[0] + height - features['Cell Altitude'].values[0] + features['Cell Building Height'].values[0] + 50
    # print(Z)
    # c = ['r']
    # ax.bar3d(X, Y, height, width, depth, Z, shade=True, alpha=1.0, zorder=4)

    # map_data = map_building + map_att

    xi, yi = np.mgrid[0:63:64j, 0:63:64j]

    surf = ax.plot_surface(xi, yi,
                           map_data,
                           # cmap=cm.coolwarm,
                           alpha=0.9,
                           # linewidth=0, antialiased=True,
                           # zorder=3
                           )

    # cset = ax.contourf(xi, yi, map_data, zdir='z', offset=100,
    #                    alpha=0.8,cmap=cm.coolwarm,
    #                    zorder=2)

    cset = ax.contourf(xi, yi, map_data, zdir='z',
                       alpha=0.8,
                       offset=0,
                       # cmap=cm.coolwarm
                       )

    c_list = (map_rsrp - map_rsrp.min())/(map_rsrp.max()-map_rsrp.min())*10

    # ax.scatter(col[0], col[1], zs=100, zdir='z',
    #            # c=c_list,
    #            s=1,
    #            alpha=0.6,
    #            zorder=1,
    #            # label='RSRP'
    #            )



    # Customize the z axis.
    # ax.set_zlim3d(0, 530)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=8)
    ax.legend()

    plt.show()






def draw_building_3D_clear(map_data, map_mask,map_rsrp, features):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)

    col = np.nonzero(map_mask)
    map_rsrp = map_rsrp[map_mask.astype('bool')]



    # 设置柱子属性
    height = 505  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 1  # 柱子的长和宽

    X = 32
    Y = 0
    Z = features['Height'].values[0] + height - features['Cell Altitude'].values[0] + features['Cell Building Height'].values[0]
    print(Z)
    c = ['r']
    ax.bar3d(X, Y, height, width, depth, Z, shade=True, alpha=1.0, zorder=4)

    # map_data = map_building + map_att

    xi, yi = np.mgrid[0:63:64j, 0:63:64j]

    surf = ax.plot_surface(xi, yi,
                           map_data,
                           cmap=cm.coolwarm,
                           alpha=0.9,
                           linewidth=0, antialiased=True,
                           zorder=3)

    cset = ax.contourf(xi, yi, map_data, zdir='z', offset=100,
                       alpha=0.8,cmap=cm.coolwarm,
                       zorder=2)

    c_list = (map_rsrp - map_rsrp.min())/(map_rsrp.max()-map_rsrp.min())*10

    # ax.scatter(col[0], col[1], zs=100, zdir='z',
    #            c=c_list,
    #            s=1,
    #            alpha=0.6,
    #            zorder=1,
    #            label='RSRP')

    # Customize the z axis.
    # ax.set_zlim3d(-200, 40)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=8)
    ax.legend()

    plt.show()

def draw_inputs_2D(data, title, map_rsrp, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    import matplotlib as mpl
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    surf = ax.imshow(data,
                     # norm=vnorm,
                     alpha=0.6,
                     cmap=cm.coolwarm)

    col = np.nonzero(map_mask)
    # map_rsrp = map_rsrp[map_mask.astype('bool')]

    if not ML:
        map_rsrp = map_rsrp[map_mask.astype('bool')]
        # rsrp = []
        # for i in range(len(col[0])):
        #     p = map_rsrp[col[1][i],col[0][i]]
        #     if p > map_rsrp_max:
        #         p = map_rsrp_max
        #     elif p < map_rsrp_min:
        #         p = map_rsrp_min
        #     rsrp.append(p)
        # map_rsrp = np.array(rsrp)

    # c_list = (map_rsrp - map_rsrp_min) / (map_rsrp_max -map_rsrp_min) * 10
    c_list = (map_rsrp - map_rsrp.min()) / (map_rsrp.max() - map_rsrp.min()) * 10
    ax.scatter(col[1], col[0],
               c=c_list,
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
    fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()



def draw_residual_2D(title, map_rsrp, map_mask, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    map_rsrp = pre_data(map_rsrp, map_mask, need_filter=False)

    import matplotlib as mpl
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    surf = ax.imshow(map_rsrp,
                     # norm=vnorm,
                     alpha=1.0,
                     # cmap=cm.coolwarm
                     )

    col = np.nonzero(map_mask)
    # xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    # map_rsrp = map_rsrp[map_mask.astype('bool')]

    # if not ML:
    #     map_rsrp = map_rsrp[map_mask.astype('bool')]
        # rsrp = []
        # for i in range(len(col[0])):
        #     p = map_rsrp[col[0][i],col[1][i]]
        #     if p > map_rsrp_max:
        #         p = map_rsrp_max
        #     elif p < map_rsrp_min:
        #         p = map_rsrp_min
        #     rsrp.append(p)
        # map_rsrp = np.array(rsrp)

    # c_list = (map_rsrp - map_rsrp_min) / (map_rsrp_max -map_rsrp_min) * 10
    # c_list = (map_rsrp - 0) / (1 - 0) * 10
    # ax.scatter(col[1], 64-col[0],
    #            c=c_list,
    #            s=10,
    #            alpha=1.0,
    #            label='Mask')


    ax.scatter(0, 32,
               color='red',
               s=100,
               marker='s',
               alpha=1.0,
               label='BS')

    # ax.set_title(title)
    # fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()



def draw_inputs_2D_clear(title, map_rsrp, map_mask, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    import matplotlib as mpl
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    # surf = ax.imshow(map_rsrp,
    #                  # norm=vnorm,
    #                  alpha=0.6,
    #                  cmap=cm.coolwarm)

    col = np.nonzero(map_mask)
    # xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    # map_rsrp = map_rsrp[map_mask.astype('bool')]

    if not ML:
        map_rsrp = map_rsrp[map_mask.astype('bool')]
        # rsrp = []
        # for i in range(len(col[0])):
        #     p = map_rsrp[col[0][i],col[1][i]]
        #     if p > map_rsrp_max:
        #         p = map_rsrp_max
        #     elif p < map_rsrp_min:
        #         p = map_rsrp_min
        #     rsrp.append(p)
        # map_rsrp = np.array(rsrp)

    # c_list = (map_rsrp - map_rsrp_min) / (map_rsrp_max -map_rsrp_min) * 10
    c_list = (map_rsrp - 0) / (1 - 0) * 10
    ax.scatter(col[1], 64 - col[0],
               c=c_list,
               s=50,
               marker='s',
               alpha=1.0,
               label='Residual')

    ax.scatter(0, 32,
               color='red',
               s=100,
               marker='s',
               alpha=1.0,
               label='BS')

    ax.set_title(title)
    # fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()

def eval_prediction(cls_true, cls_pred):
    y_true = []
    for ct in cls_true:
        if ct < -115:
            y_true.append(0)
        elif (ct >= -115) & (ct < -105):
            y_true.append(1)
        elif (ct >= -105) & (ct < -95):
            y_true.append(2)
        elif (ct >= -95) & (ct < -85):
            y_true.append(3)
        else:
            y_true.append(4)

    y_pred = []
    for cp in cls_pred:
        if cp < -115:
            y_pred.append(0)
        elif (cp >= -115) & (cp < -105):
            y_pred.append(1)
        elif (cp >= -105) & (cp < -95):
            y_pred.append(2)
        elif (cp >= -95) & (cp < -85):
            y_pred.append(3)
        else:
            y_pred.append(4)
    # print(y_true)
    # print(y_pred)
    predicitons = (np.array(y_true) == np.array(y_pred))
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    #
    # # precision = precision_score(y_true, y_pred, average='macro')
    # precision = precision_score(y_true, y_pred, average='weighted')
    # accuracy = accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted')
    return predicitons
def draw_T_F_2D(data, title, map_rsrp, true_rsrp, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    import matplotlib as mpl
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    surf = ax.imshow(data,
                     # norm=vnorm,
                     alpha=0.6,
                     cmap=cm.coolwarm)

    col = np.nonzero(map_mask)
    if not ML:
        map_rsrp = map_rsrp[map_mask.astype('bool')]
    true_rsrp = true_rsrp[map_mask.astype('bool')]

    # rsrp = []
    # for i in range(len(col[0])):
    #     p = map_rsrp[col[0][i], col[1][i]]
    #     if p > map_rsrp_max:
    #         p = map_rsrp_max
    #     elif p < map_rsrp_min:
    #         p = map_rsrp_min
    #     rsrp.append(p)
    # map_rsrp = np.array(rsrp)

    preditions = eval_prediction(map_rsrp, true_rsrp)
    print(preditions)

    c_list = []
    for p in preditions:
        if p:
           c_list.append('g')
        else:
            c_list.append('r')
    # c_list = preditions
    ax.scatter(col[1], col[0],
               c=c_list,
               s=100,
               marker='x',
               alpha=1.0,
               label='RSRP')

    ax.scatter(0, 32,
               color='red',
               s=100,
               marker='s',
               alpha=1.0,
               label='BS')

    ax.set_title(title)
    fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()

def get_cost_features(features):
    att_min = features['Altitude'].min()

    f = features['Frequency Band'].values

    h_bs = features['Cell Altitude'].values + \
           features['Cell Building Height'].values + \
           features['Height'].values - att_min
    h_ue = features['Altitude'].values \
           - att_min + 1.0
    ci = features['Cell Clutter Index']

    x = features['X'] - features['Cell X']
    y = features['Y'] - features['Cell Y']
    d = np.sqrt(np.square(x) + np.square(y)).values / 1000

    # print("f:", f[0], "d:", d[0], "h_bs:", h_bs[0], "h_ue:", h_ue[0])

    return f, d, h_bs, h_ue, ci
def get_map(features, pred_RSRP, w,h):
    x_ = np.array(features['x_'].values).astype(np.int) + w / 2
    y_ = np.array(features['y_'].values).astype(np.int)
    Hm = features['Building Height'].values

    map_rsrp_pred = np.zeros([w, h])
    map_building_mask = np.zeros([w, h])
    for i in range(len(x_)):
        if x_[i] < w:
            if y_[i] < h:
                xx = int(x_[i])
                yy = int(y_[i])
                map_rsrp_pred[xx, yy] = pred_RSRP[i]
                # map_rsrp_pred[xx, yy] = 1
                if Hm[i] != 0:
                    map_building_mask[xx, yy] = 1
    return map_rsrp_pred, map_building_mask
def SPM(features, w, h):
    K1, K2, K3, K4, K5, K6 = 25.46, 36.24, 17.57, 0, -9.9, 0

    K_clutter = 1

    Diffraction = 0.2

    f, d, h_bs, h_ue, ci = get_cost_features(features)
    d = d*1000
    pl = []
    for i in range(len(f)):
        PL = K1+K2*np.log10(d[i])+K3*np.log10(h_bs[i])+K4*Diffraction+\
        K5*np.log10(h_bs[i])*np.log10(d[i])+K6*h_ue[i]+K_clutter
        # PL = 150

        pl.append(PL)
    # print("pl：", pl[0])
    RSP = features['RS Power'].values
    RSRP = features['RSRP'].values

    pl = np.array(pl)

    pred_RSRP = RSP - pl

    # print("pred_RSRP:", np.mean(pred_RSRP), "RSRP:", np.mean(RSRP), np.shape(pred_RSRP))
    map_rsrp_pred, map_building_mask = get_map(features, pred_RSRP, w, h)

    map_rsrp_pred = map_rsrp_pred - (map_building_mask * 3.365)

    # if cnt == 4:
    #     generate_images(map_rsrp_pred, map_rsrp, map_deltaH, 'SPM')

    return map_rsrp_pred

def maxmin_norm(data, mask):
    data = (data-data[mask.astype('bool')].min()) \
                 / (data[mask.astype('bool')].max() - data[mask.astype('bool')].min())
    data = data * mask
    return data

def GAN_model(features,model_dir,fine_turning=False):
    from Features2img import get_map_features
    from gd_rsrp import Discriminator, Generator
    generator = Generator()
    # generator.load_weights("./model/GAN/generator")

    generator.load_weights(model_dir)


    map_deltaH, map_mask, map_pl, map_pl_pred, map_building = get_map_features(features, weight, height)

    # #normlization
    map_deltaH = maxmin_norm(map_deltaH, map_mask)
    # train_data = [map_deltaH, map_mask, map_pl, map_pl_pred]
    rsp = features['RS Power'].values[0]

    batch_size = 1
    input_image = np.reshape(map_deltaH, [1, weight, height, 1])


    # target = test_data[i][2]

    mask = np.reshape(map_mask, [batch_size, weight, height, 1])
    map_building = np.reshape(map_building, [batch_size, weight, height, 1])
    # target = np.reshape(target, [batch_size, weight, height, 1])

    ground_ture = map_pl
    fitting_pred = map_pl_pred
    ground_ture = np.reshape(ground_ture, [batch_size, weight, height, 1])
    fitting_pred = np.reshape(fitting_pred, [batch_size, weight, height, 1])

    target = np.reshape(ground_ture - fitting_pred , [batch_size, weight, height, 1])

    # print("target max:", np.max(target), "target min:", np.min(target),)

    prediction = generator(input_image, training=False)
    prediction = prediction * (target[mask.astype('bool')].max() - target[mask.astype('bool')].min()) \
                 + target[mask.astype('bool')].min()
    prediction = prediction * mask

    if fine_turning:
        pred_PL = (prediction + fitting_pred) * mask
    else:
        pred_PL = (prediction + fitting_pred - (~map_building.astype('bool') * 3.365)) * mask
    pred_RSRP = (rsp - pred_PL) * mask

    return np.reshape(pred_RSRP,[weight, height, 1])


def GAN_model_org(features,model_dir,fine_turning=False):
    from Features2img import get_map_features
    from gd_rsrp import Discriminator, Generator
    generator = Generator()
    # generator.load_weights("./model/GAN/generator")

    generator.load_weights(model_dir)


    map_deltaH, map_mask, map_pl, map_pl_pred, map_building = get_map_features(features, weight, height)

    # #normlization
    map_deltaH = maxmin_norm(map_deltaH, map_mask)
    # train_data = [map_deltaH, map_mask, map_pl, map_pl_pred]
    rsp = features['RS Power'].values[0]

    batch_size = 1
    input_image = np.reshape(map_deltaH, [1, weight, height, 1])


    # target = test_data[i][2]

    mask = np.reshape(map_mask, [batch_size, weight, height, 1])
    map_building = np.reshape(map_building, [batch_size, weight, height, 1])
    # target = np.reshape(target, [batch_size, weight, height, 1])

    ground_ture = map_pl
    fitting_pred = map_pl_pred
    ground_ture = np.reshape(ground_ture, [batch_size, weight, height, 1])
    fitting_pred = np.reshape(fitting_pred, [batch_size, weight, height, 1])

    target = np.reshape(ground_ture - fitting_pred , [batch_size, weight, height, 1])

    # print("target max:", np.max(target), "target min:", np.min(target),)

    prediction = generator(input_image, training=False)
    prediction = prediction * (target[mask.astype('bool')].max() - target[mask.astype('bool')].min()) \
                 + target[mask.astype('bool')].min()
    # prediction = prediction * mask

    # if fine_turning:
    #     pred_PL = (prediction + fitting_pred) * mask
    # else:
    #     pred_PL = (prediction + fitting_pred - (~map_building.astype('bool') * 3.365)) * mask
    # pred_RSRP = (rsp - pred_PL) * mask

    return prediction

def get_ML_features(features):
    att_min = features['Altitude'].min()

    f = features['Frequency Band'].values

    h_bs = features['Cell Altitude'].values + \
           features['Cell Building Height'].values + \
           features['Height'].values - att_min
    h_ue = features['Altitude'].values \
           - att_min + 1.0

    x = features['X'] - features['Cell X']
    y = features['Y'] - features['Cell Y']
    d = np.sqrt(np.square(x) + np.square(y)).values

    # 3.垂直下倾角
    betaV = features['Electrical Downtilt'].values[0] + features['Mechanical Downtilt'].values[0]

    # 8.用户建筑物高度m
    Hm = features['Building Height'].values
    # 9.有效高度m
    deltaH = h_bs - h_ue - Hm
    # 16.信号线与用户栅格建筑顶的相对高度m
    deltaHv = deltaH - (d * np.tan(np.deg2rad(betaV)))

    # print("f:", f[0], "d:", d[0], "h_bs:", h_bs[0], "h_ue:", h_ue[0])
    rsrp = features['RSRP'].values

    pl = features['RS Power'].values - rsrp

    w, h = 64, 64
    x_ = np.array(features['x_'].values).astype(np.int) + w / 2
    y_ = np.array(features['y_'].values).astype(np.int)
    # x_ = np.floor(pb_data['x_'].values + w/2, 0)
    # y_ = np.floor(pb_data['y_'].values, 0)

    # print(len(x_), len(y_))

    map_deltaH = np.zeros([w, h])
    map_mask = np.zeros([w, h])
    map_pl = np.zeros([w, h])
    map_d = np.zeros([w, h])
    map_pl_pred = np.zeros([w, h])
    map_deltaHv = np.zeros([w, h])
    map_hm = np.zeros([w, h])
    map_att = np.zeros([w, h])
    # deltaH = (h_bs - h_ue + hm)
    # print("deltaH:", len(deltaH))
    # print("RSRP.mean:", RSRP.mean())

    for i in range(len(x_)):
        # print(i)
        # print(deltaH[i])
        # print(int(x_[i]), int(y_[i]), int(x[i]), int(y[i]))
        if x_[i] < w:
            if y_[i] < h:

                xx = int(x_[i])
                yy = int(y_[i])
                map_mask[xx, yy] = 1
                # map_rsrp[xx, yy] = (RSRP.max()-RSRP[i])/(RSRP.max()-RSRP.min())
                # map_deltaH[xx, yy] = (deltaH.max()-deltaH[i])/(deltaH.max()-deltaH.min())
                # map_deltaHv[xx, yy] = (deltaHv.max() - deltaHv[i]) / (deltaHv.max() - deltaHv.min())

                map_d[xx, yy] = d[i]
                map_hm[xx, yy] = Hm[i]
                map_att[xx, yy] = h_ue[i]

                map_pl[xx, yy] = pl[i]
                # map_RSP[xx, yy] = RSP[i]

                map_deltaHv[xx, yy] = deltaHv[i]
                map_deltaH[xx, yy] = deltaH[i]


    # log_d = np.log10(d)
    # log_h_bs = np.log10(h_bs)
    # log_h_ue = np.log10(h_ue)
    return map_d[map_mask.astype('bool')], map_hm[map_mask.astype('bool')],\
           map_att[map_mask.astype('bool')],\
           map_deltaH[map_mask.astype('bool')],\
           map_deltaHv[map_mask.astype('bool')],\
           map_pl[map_mask.astype('bool')]
def ML_model(features, path):
    # X_test, Y_test = get_dataset("./dataset/csv_set_h50_64/testset", is_training=False)
    d, Hm, h_ue, deltaH, deltaHv, PL = get_ML_features(features)


    rsp = features['RS Power'].values
    # 8.用户建筑物高度m
    # Hm = features['Building Height'].values

    # pfs = np.array([d, h_bs, h_ue,
    #                     h_bs*d, deltaH, betaV])
    pfs = np.array([d, Hm, h_ue,
                    deltaH, deltaHv])
    print("shape:", np.shape(pfs))

    # if is_training:
    #     PL = (PL-61.2) / 90.9
    # else:
    #     PL = (PL-60.2) / 88.9

    X_test = pfs.T

    import pickle
    with open(path, 'rb') as fr:
        model = pickle.load(fr)

        test_RMSE, test_MAPE = 0, 0
        test_cnt = 0
        # precision, accuracy,f1 = 0,0,0
        # for i in range(len(X_test)):
        test_cnt += 1

        rsp = rsp[0]
        print("rsp shape:", np.shape(rsp), rsp)
        # X_test = X_test[:, :-1]
        print("X_test shape:", np.shape(X_test))
        # print(X_test[:10])

        # X_test = X_test[:,:]
        # print("X_test shape:",np.shape(X_test))

        prediction = model.predict(X_test)
        print("PREDICTION shape:", np.shape(prediction))

    return rsp - prediction

def get_deltaHv(features, map_data):
    # 3.垂直下倾角
    betaV = features['Electrical Downtilt'].values[0] + features['Mechanical Downtilt'].values[0]
    # 6.基站塔顶高度m
    Hb = features['Height'].values[0] + features['Cell Building Height'].values[0] + features['Cell Altitude'].values[0]+10

    L = np.zeros([64,64])
    for i in range(64):
        for j in range(64):
            l = np.sqrt(np.square(i-32) + np.square(j-0))
            L[i,j] = l
    L = np.array(L)
            # 16.信号线与用户栅格建筑顶的相对高度m
    print(betaV, Hb)
    deltaHv = Hb - (L * np.tan(np.deg2rad(betaV+5))) - map_data
    return deltaHv

import seaborn as sns
def fig_range(RSRP_map):
    data = RSRP_map[map_mask.astype('bool')]
    # data = np.ndarray.flatten(RSRP_map)
    print(np.shape(data),data)

    plt.figure(figsize=(3.8, 1.5))
    sns.distplot(data)
    plt.xlabel("RSRP range")
    plt.show()

def mape(x1,x2):
    # return (x1-x2)/(x2)
    return (x1 - x2)

def loss_show(pred_RSRP_map, RSRP_map, title):
    true = RSRP_map[map_mask.astype('bool')]
    N = len(pred_RSRP_map)
    ig, ax = plt.subplots(N, 1)
    # preds = []
    for i in range(N):
        if len(pred_RSRP_map[i]) != len(true):
            tmp = mape(pred_RSRP_map[i][map_mask.astype('bool')], true)
        else:
            tmp = mape(pred_RSRP_map[i], true)
        # preds.append(tmp)

        sns.distplot(tmp,ax=ax[i])
        ax[i].set_title(title[i])
    plt.show()



    # data = np.ndarray.flatten(RSRP_map)
    # print(np.shape(data),data)



data_path = "./dataset/csv_set_h50_64/testset/"

test_filenames = cls_ult.get_filename("./dataset/csv_set_h50_64/testset/")

cnt=0
for name, context in test_filenames:
    cnt+=1
    # print(name,context)
    features = pd.read_csv(context)
    map_mask, map_att, map_building, map_deltaHv, map_ci, map_rsrp = get_draw_map(features, 64, 64)
    map_building = pre_data(map_building, map_mask)
    map_att = pre_data(map_att, map_mask,need_filter=False)
    map_data = map_building + map_att


    # deltaHv = get_deltaHv(features, map_data)
    # deltaHv = maxmin_norm(deltaHv,map_mask)

    draw_building_3D(map_data, map_mask, map_rsrp, features)

    # map_rsrp_pred_spm = SPM(features,64,64)

    # draw_inputs_2D(map_data, 'Grand truth on topographic projection', map_rsrp)
    # draw_inputs_2D_clear(deltaHv, 'Topographic projection of deltaHv', map_rsrp)

    # map_rsrp_pred1 = GAN_model(features, "./model/GAN_SC_1/g19")
    # map_rsrp_pred2 = GAN_model(features, "./model/UNET_SC/g55")
    # map_rsrp_pred_gan = GAN_model(features, "./model/GAN_SC_1/g91",fine_turning=True)

    # map_rsrp_pred = GAN_model_org(features, "./model/GAN_SC/g82")

    # map_rsrp_pred = np.reshape(map_rsrp_pred, [64, 64])
    # print(np.shape(map_rsrp_pred), np.shape(map_rsrp))

    # map_residual = pre_data(map_rsrp-map_rsrp_pred_spm, map_mask, need_filter=False)
    # map_rsrp_pred_spm = pre_data(map_rsrp_pred_spm, map_mask)
    # draw_residual_2D('Ground Truth', map_rsrp, map_mask)
    # draw_residual_2D('Ground Truth', map_rsrp_pred_spm, map_mask)

    # draw_residual_2D('Predicted Residual', map_residual, map_mask)
    # draw_inputs_2D_clear('Predicted by SPM', map_rsrp_pred_spm, map_mask)
    # draw_inputs_2D(map_data, 'Prediction results by GAN on topographic projection', map_rsrp_pred_gan)


    # draw_inputs_2D(map_data, 'Prediction results by GAN on topographic projection', map_rsrp_pred)

    # draw_T_F_2D(map_data, 'Prediction results by GAN', map_rsrp_pred_gan, true_rsrp = map_rsrp)

    # ML_pred1 = ML_model(features, path='./model/ML/rfr_sc.pickle')
    # ML_pred2 = ML_model(features, path='./model/ML/knn_sc.pickle')
    # # print("ML_pred shape:", np.shape(ML_pred))
    # draw_inputs_2D(map_data, 'Prediction results by rfr on topographic projection', ML_pred, ML=True)
    #
    # draw_T_F_2D(map_data, 'Prediction results by RFR', ML_pred, true_rsrp=map_rsrp, ML=True)

    # fig_range(map_rsrp-map_rsrp_pred_spm)
    # loss_show([map_rsrp_pred_spm,
    #            ML_pred2,
    #            ML_pred1,
    #            ],
    #           map_rsrp,
    #           ['Loss of SPM',
    #            'Loss of KNN',
    #            'Loss of GAN',
    #            ])

    if cnt == 1:
        break

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import cls_ult
weight, height = 64,64

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

from sklearn.impute import KNNImputer
def img_imputer(input_image, mask):
    input_image = np.array(np.reshape(input_image, [weight, height]))
    mask = np.reshape(mask, [weight, height])
    input_image[~mask.astype('bool')] = np.nan
    # input_image = np.nan_to_num(input_image)
    # print(np.shape(input_image), np.mean(input_image))
    # print("Before: ", np.shape(input_image))
    imputer = KNNImputer(n_neighbors=1)
    output_image = imputer.fit_transform(input_image)
    # print("After: ", np.shape(output_image))
    # output_image = np.pad(output_image,((0,weight-np.shape(output_image)[0]),(0,height-np.shape(output_image)[1])),'constant',constant_values=1)
    # print("After that: ", np.shape(output_image))
    # output_image = np.reshape(output_image,[batch_size,weight,height,input_image_channel])
    return output_image

def pre_data(map_data, map_mask, need_filter=True):
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

def draw_building_3D(map_data, map_mask,map_rsrp, features):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)

    col = np.nonzero(map_mask)
    map_rsrp = map_rsrp[map_mask.astype('bool')]



    # 设置柱子属性
    height = map_data.max()  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 1  # 柱子的长和宽

    # X = 32
    # Y = 0
    # Z = features['Height'].values[0] + height - features['Cell Altitude'].values[0] + features['Cell Building Height'].values[0] + 50
    # print(Z)
    # c = ['r']
    # ax.bar3d(X, Y, height, width, depth, Z, shade=True, alpha=1.0, zorder=4)

    # map_data = map_building + map_att

    xi, yi = np.mgrid[0:63:64j, 0:63:64j]

    surf = ax.plot_surface(xi, yi,
                           map_data,
                           # cmap=cm.coolwarm,
                           alpha=0.9,
                           # linewidth=0, antialiased=True,
                           # zorder=3
                           )

    # cset = ax.contourf(xi, yi, map_data, zdir='z', offset=100,
    #                    alpha=0.8,cmap=cm.coolwarm,
    #                    zorder=2)

    cset = ax.contourf(xi, yi, map_data, zdir='z',
                       alpha=0.8,
                       offset=0,
                       # cmap=cm.coolwarm
                       )

    c_list = (map_rsrp - map_rsrp.min())/(map_rsrp.max()-map_rsrp.min())*10

    # ax.scatter(col[0], col[1], zs=100, zdir='z',
    #            # c=c_list,
    #            s=1,
    #            alpha=0.6,
    #            zorder=1,
    #            # label='RSRP'
    #            )



    # Customize the z axis.
    ax.set_zlim3d(0, 530)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=8)
    # ax.legend()

    plt.show()






def draw_building_3D_clear(map_data, map_mask,map_rsrp, features):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)

    col = np.nonzero(map_mask)
    map_rsrp = map_rsrp[map_mask.astype('bool')]



    # 设置柱子属性
    height = 505  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 1  # 柱子的长和宽

    X = 32
    Y = 0
    Z = features['Height'].values[0] + height - features['Cell Altitude'].values[0] + features['Cell Building Height'].values[0]
    print(Z)
    c = ['r']
    ax.bar3d(X, Y, height, width, depth, Z, shade=True, alpha=1.0, zorder=4)

    # map_data = map_building + map_att

    xi, yi = np.mgrid[0:63:64j, 0:63:64j]

    surf = ax.plot_surface(xi, yi,
                           map_data,
                           cmap=cm.coolwarm,
                           alpha=0.9,
                           linewidth=0, antialiased=True,
                           zorder=3)

    cset = ax.contourf(xi, yi, map_data, zdir='z', offset=100,
                       alpha=0.8,cmap=cm.coolwarm,
                       zorder=2)

    c_list = (map_rsrp - map_rsrp.min())/(map_rsrp.max()-map_rsrp.min())*10

    # ax.scatter(col[0], col[1], zs=100, zdir='z',
    #            c=c_list,
    #            s=1,
    #            alpha=0.6,
    #            zorder=1,
    #            label='RSRP')

    # Customize the z axis.
    # ax.set_zlim3d(-200, 40)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=8)
    ax.legend()

    plt.show()

def draw_inputs_2D(data, title, map_rsrp, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    import matplotlib as mpl
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    surf = ax.imshow(data,
                     # norm=vnorm,
                     alpha=0.6,
                     cmap=cm.coolwarm)

    col = np.nonzero(map_mask)
    # map_rsrp = map_rsrp[map_mask.astype('bool')]

    if not ML:
        map_rsrp = map_rsrp[map_mask.astype('bool')]
        # rsrp = []
        # for i in range(len(col[0])):
        #     p = map_rsrp[col[1][i],col[0][i]]
        #     if p > map_rsrp_max:
        #         p = map_rsrp_max
        #     elif p < map_rsrp_min:
        #         p = map_rsrp_min
        #     rsrp.append(p)
        # map_rsrp = np.array(rsrp)

    # c_list = (map_rsrp - map_rsrp_min) / (map_rsrp_max -map_rsrp_min) * 10
    c_list = (map_rsrp - map_rsrp.min()) / (map_rsrp.max() - map_rsrp.min()) * 10
    ax.scatter(col[1], col[0],
               c=c_list,
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
    fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()



def draw_residual_2D(title, map_rsrp, map_mask, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    map_rsrp = pre_data(map_rsrp, map_mask, need_filter=False)

    import matplotlib as mpl
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    surf = ax.imshow(map_rsrp,
                     # norm=vnorm,
                     alpha=1.0,
                     # cmap=cm.coolwarm
                     )

    col = np.nonzero(map_mask)
    # xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    # map_rsrp = map_rsrp[map_mask.astype('bool')]

    # if not ML:
    #     map_rsrp = map_rsrp[map_mask.astype('bool')]
        # rsrp = []
        # for i in range(len(col[0])):
        #     p = map_rsrp[col[0][i],col[1][i]]
        #     if p > map_rsrp_max:
        #         p = map_rsrp_max
        #     elif p < map_rsrp_min:
        #         p = map_rsrp_min
        #     rsrp.append(p)
        # map_rsrp = np.array(rsrp)

    # c_list = (map_rsrp - map_rsrp_min) / (map_rsrp_max -map_rsrp_min) * 10
    # c_list = (map_rsrp - 0) / (1 - 0) * 10
    # ax.scatter(col[1], 64-col[0],
    #            c=c_list,
    #            s=10,
    #            alpha=1.0,
    #            label='Mask')


    ax.scatter(0, 32,
               color='red',
               s=100,
               marker='s',
               alpha=1.0,
               label='BS')

    # ax.set_title(title)
    # fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()



def draw_inputs_2D_clear(title, map_rsrp, map_mask, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    import matplotlib as mpl
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    # surf = ax.imshow(map_rsrp,
    #                  # norm=vnorm,
    #                  alpha=0.6,
    #                  cmap=cm.coolwarm)

    col = np.nonzero(map_mask)
    # xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    # map_rsrp = map_rsrp[map_mask.astype('bool')]

    if not ML:
        map_rsrp = map_rsrp[map_mask.astype('bool')]
        # rsrp = []
        # for i in range(len(col[0])):
        #     p = map_rsrp[col[0][i],col[1][i]]
        #     if p > map_rsrp_max:
        #         p = map_rsrp_max
        #     elif p < map_rsrp_min:
        #         p = map_rsrp_min
        #     rsrp.append(p)
        # map_rsrp = np.array(rsrp)

    # c_list = (map_rsrp - map_rsrp_min) / (map_rsrp_max -map_rsrp_min) * 10
    c_list = (map_rsrp - 0) / (1 - 0) * 10
    ax.scatter(col[1], 64 - col[0],
               c=c_list,
               s=50,
               marker='s',
               alpha=1.0,
               label='Residual')

    ax.scatter(0, 32,
               color='red',
               s=100,
               marker='s',
               alpha=1.0,
               label='BS')

    ax.set_title(title)
    # fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()

def eval_prediction(cls_true, cls_pred):
    y_true = []
    for ct in cls_true:
        if ct < -115:
            y_true.append(0)
        elif (ct >= -115) & (ct < -105):
            y_true.append(1)
        elif (ct >= -105) & (ct < -95):
            y_true.append(2)
        elif (ct >= -95) & (ct < -85):
            y_true.append(3)
        else:
            y_true.append(4)

    y_pred = []
    for cp in cls_pred:
        if cp < -115:
            y_pred.append(0)
        elif (cp >= -115) & (cp < -105):
            y_pred.append(1)
        elif (cp >= -105) & (cp < -95):
            y_pred.append(2)
        elif (cp >= -95) & (cp < -85):
            y_pred.append(3)
        else:
            y_pred.append(4)
    # print(y_true)
    # print(y_pred)
    predicitons = (np.array(y_true) == np.array(y_pred))
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    #
    # # precision = precision_score(y_true, y_pred, average='macro')
    # precision = precision_score(y_true, y_pred, average='weighted')
    # accuracy = accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted')
    return predicitons
def draw_T_F_2D(data, title, map_rsrp, true_rsrp, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    import matplotlib as mpl
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    surf = ax.imshow(data,
                     # norm=vnorm,
                     alpha=0.6,
                     cmap=cm.coolwarm)

    col = np.nonzero(map_mask)
    if not ML:
        map_rsrp = map_rsrp[map_mask.astype('bool')]
    true_rsrp = true_rsrp[map_mask.astype('bool')]

    # rsrp = []
    # for i in range(len(col[0])):
    #     p = map_rsrp[col[0][i], col[1][i]]
    #     if p > map_rsrp_max:
    #         p = map_rsrp_max
    #     elif p < map_rsrp_min:
    #         p = map_rsrp_min
    #     rsrp.append(p)
    # map_rsrp = np.array(rsrp)

    preditions = eval_prediction(map_rsrp, true_rsrp)
    print(preditions)

    c_list = []
    for p in preditions:
        if p:
           c_list.append('g')
        else:
            c_list.append('r')
    # c_list = preditions
    ax.scatter(col[1], col[0],
               c=c_list,
               s=100,
               marker='x',
               alpha=1.0,
               label='RSRP')

    ax.scatter(0, 32,
               color='red',
               s=100,
               marker='s',
               alpha=1.0,
               label='BS')

    ax.set_title(title)
    fig.colorbar(surf, fraction=0.05, pad=0.05)
    ax.legend()
    # plt.savefig('images/epoch'+str(epoch)+'_'+str(step)+'.png')
    # print('saved images.')
    plt.show()

def get_cost_features(features):
    att_min = features['Altitude'].min()

    f = features['Frequency Band'].values

    h_bs = features['Cell Altitude'].values + \
           features['Cell Building Height'].values + \
           features['Height'].values - att_min
    h_ue = features['Altitude'].values \
           - att_min + 1.0
    ci = features['Cell Clutter Index']

    x = features['X'] - features['Cell X']
    y = features['Y'] - features['Cell Y']
    d = np.sqrt(np.square(x) + np.square(y)).values / 1000

    # print("f:", f[0], "d:", d[0], "h_bs:", h_bs[0], "h_ue:", h_ue[0])

    return f, d, h_bs, h_ue, ci
def get_map(features, pred_RSRP, w,h):
    x_ = np.array(features['x_'].values).astype(np.int) + w / 2
    y_ = np.array(features['y_'].values).astype(np.int)
    Hm = features['Building Height'].values

    map_rsrp_pred = np.zeros([w, h])
    map_building_mask = np.zeros([w, h])
    for i in range(len(x_)):
        if x_[i] < w:
            if y_[i] < h:
                xx = int(x_[i])
                yy = int(y_[i])
                map_rsrp_pred[xx, yy] = pred_RSRP[i]
                # map_rsrp_pred[xx, yy] = 1
                if Hm[i] != 0:
                    map_building_mask[xx, yy] = 1
    return map_rsrp_pred, map_building_mask
def SPM(features, w, h):
    K1, K2, K3, K4, K5, K6 = 25.46, 36.24, 17.57, 0, -9.9, 0

    K_clutter = 1

    Diffraction = 0.2

    f, d, h_bs, h_ue, ci = get_cost_features(features)
    d = d*1000
    pl = []
    for i in range(len(f)):
        PL = K1+K2*np.log10(d[i])+K3*np.log10(h_bs[i])+K4*Diffraction+\
        K5*np.log10(h_bs[i])*np.log10(d[i])+K6*h_ue[i]+K_clutter
        # PL = 150

        pl.append(PL)
    # print("pl：", pl[0])
    RSP = features['RS Power'].values
    RSRP = features['RSRP'].values

    pl = np.array(pl)

    pred_RSRP = RSP - pl

    # print("pred_RSRP:", np.mean(pred_RSRP), "RSRP:", np.mean(RSRP), np.shape(pred_RSRP))
    map_rsrp_pred, map_building_mask = get_map(features, pred_RSRP, w, h)

    map_rsrp_pred = map_rsrp_pred - (map_building_mask * 3.365)

    # if cnt == 4:
    #     generate_images(map_rsrp_pred, map_rsrp, map_deltaH, 'SPM')

    return map_rsrp_pred

def maxmin_norm(data, mask):
    data = (data-data[mask.astype('bool')].min()) \
                 / (data[mask.astype('bool')].max() - data[mask.astype('bool')].min())
    data = data * mask
    return data

def GAN_model(features,model_dir,fine_turning=False):
    from Features2img import get_map_features
    from gd_rsrp import Discriminator, Generator
    generator = Generator()
    # generator.load_weights("./model/GAN/generator")

    generator.load_weights(model_dir)


    map_deltaH, map_mask, map_pl, map_pl_pred, map_building = get_map_features(features, weight, height)

    # #normlization
    map_deltaH = maxmin_norm(map_deltaH, map_mask)
    # train_data = [map_deltaH, map_mask, map_pl, map_pl_pred]
    rsp = features['RS Power'].values[0]

    batch_size = 1
    input_image = np.reshape(map_deltaH, [1, weight, height, 1])


    # target = test_data[i][2]

    mask = np.reshape(map_mask, [batch_size, weight, height, 1])
    map_building = np.reshape(map_building, [batch_size, weight, height, 1])
    # target = np.reshape(target, [batch_size, weight, height, 1])

    ground_ture = map_pl
    fitting_pred = map_pl_pred
    ground_ture = np.reshape(ground_ture, [batch_size, weight, height, 1])
    fitting_pred = np.reshape(fitting_pred, [batch_size, weight, height, 1])

    target = np.reshape(ground_ture - fitting_pred , [batch_size, weight, height, 1])

    # print("target max:", np.max(target), "target min:", np.min(target),)

    prediction = generator(input_image, training=False)
    prediction = prediction * (target[mask.astype('bool')].max() - target[mask.astype('bool')].min()) \
                 + target[mask.astype('bool')].min()
    prediction = prediction * mask

    if fine_turning:
        pred_PL = (prediction + fitting_pred) * mask
    else:
        pred_PL = (prediction + fitting_pred - (~map_building.astype('bool') * 3.365)) * mask
    pred_RSRP = (rsp - pred_PL) * mask

    return np.reshape(pred_RSRP,[weight, height, 1])


def GAN_model_org(features,model_dir,fine_turning=False):
    from Features2img import get_map_features
    from gd_rsrp import Discriminator, Generator
    generator = Generator()
    # generator.load_weights("./model/GAN/generator")

    generator.load_weights(model_dir)


    map_deltaH, map_mask, map_pl, map_pl_pred, map_building = get_map_features(features, weight, height)

    # #normlization
    map_deltaH = maxmin_norm(map_deltaH, map_mask)
    # train_data = [map_deltaH, map_mask, map_pl, map_pl_pred]
    rsp = features['RS Power'].values[0]

    batch_size = 1
    input_image = np.reshape(map_deltaH, [1, weight, height, 1])


    # target = test_data[i][2]

    mask = np.reshape(map_mask, [batch_size, weight, height, 1])
    map_building = np.reshape(map_building, [batch_size, weight, height, 1])
    # target = np.reshape(target, [batch_size, weight, height, 1])

    ground_ture = map_pl
    fitting_pred = map_pl_pred
    ground_ture = np.reshape(ground_ture, [batch_size, weight, height, 1])
    fitting_pred = np.reshape(fitting_pred, [batch_size, weight, height, 1])

    target = np.reshape(ground_ture - fitting_pred , [batch_size, weight, height, 1])

    # print("target max:", np.max(target), "target min:", np.min(target),)

    prediction = generator(input_image, training=False)
    prediction = prediction * (target[mask.astype('bool')].max() - target[mask.astype('bool')].min()) \
                 + target[mask.astype('bool')].min()
    # prediction = prediction * mask

    # if fine_turning:
    #     pred_PL = (prediction + fitting_pred) * mask
    # else:
    #     pred_PL = (prediction + fitting_pred - (~map_building.astype('bool') * 3.365)) * mask
    # pred_RSRP = (rsp - pred_PL) * mask

    return prediction

def get_ML_features(features):
    att_min = features['Altitude'].min()

    f = features['Frequency Band'].values

    h_bs = features['Cell Altitude'].values + \
           features['Cell Building Height'].values + \
           features['Height'].values - att_min
    h_ue = features['Altitude'].values \
           - att_min + 1.0

    x = features['X'] - features['Cell X']
    y = features['Y'] - features['Cell Y']
    d = np.sqrt(np.square(x) + np.square(y)).values

    # 3.垂直下倾角
    betaV = features['Electrical Downtilt'].values[0] + features['Mechanical Downtilt'].values[0]

    # 8.用户建筑物高度m
    Hm = features['Building Height'].values
    # 9.有效高度m
    deltaH = h_bs - h_ue - Hm
    # 16.信号线与用户栅格建筑顶的相对高度m
    deltaHv = deltaH - (d * np.tan(np.deg2rad(betaV)))

    # print("f:", f[0], "d:", d[0], "h_bs:", h_bs[0], "h_ue:", h_ue[0])
    rsrp = features['RSRP'].values

    pl = features['RS Power'].values - rsrp

    w, h = 64, 64
    x_ = np.array(features['x_'].values).astype(np.int) + w / 2
    y_ = np.array(features['y_'].values).astype(np.int)
    # x_ = np.floor(pb_data['x_'].values + w/2, 0)
    # y_ = np.floor(pb_data['y_'].values, 0)

    # print(len(x_), len(y_))

    map_deltaH = np.zeros([w, h])
    map_mask = np.zeros([w, h])
    map_pl = np.zeros([w, h])
    map_d = np.zeros([w, h])
    map_pl_pred = np.zeros([w, h])
    map_deltaHv = np.zeros([w, h])
    map_hm = np.zeros([w, h])
    map_att = np.zeros([w, h])
    # deltaH = (h_bs - h_ue + hm)
    # print("deltaH:", len(deltaH))
    # print("RSRP.mean:", RSRP.mean())

    for i in range(len(x_)):
        # print(i)
        # print(deltaH[i])
        # print(int(x_[i]), int(y_[i]), int(x[i]), int(y[i]))
        if x_[i] < w:
            if y_[i] < h:

                xx = int(x_[i])
                yy = int(y_[i])
                map_mask[xx, yy] = 1
                # map_rsrp[xx, yy] = (RSRP.max()-RSRP[i])/(RSRP.max()-RSRP.min())
                # map_deltaH[xx, yy] = (deltaH.max()-deltaH[i])/(deltaH.max()-deltaH.min())
                # map_deltaHv[xx, yy] = (deltaHv.max() - deltaHv[i]) / (deltaHv.max() - deltaHv.min())

                map_d[xx, yy] = d[i]
                map_hm[xx, yy] = Hm[i]
                map_att[xx, yy] = h_ue[i]

                map_pl[xx, yy] = pl[i]
                # map_RSP[xx, yy] = RSP[i]

                map_deltaHv[xx, yy] = deltaHv[i]
                map_deltaH[xx, yy] = deltaH[i]


    # log_d = np.log10(d)
    # log_h_bs = np.log10(h_bs)
    # log_h_ue = np.log10(h_ue)
    return map_d[map_mask.astype('bool')], map_hm[map_mask.astype('bool')],\
           map_att[map_mask.astype('bool')],\
           map_deltaH[map_mask.astype('bool')],\
           map_deltaHv[map_mask.astype('bool')],\
           map_pl[map_mask.astype('bool')]
def ML_model(features, path):
    # X_test, Y_test = get_dataset("./dataset/csv_set_h50_64/testset", is_training=False)
    d, Hm, h_ue, deltaH, deltaHv, PL = get_ML_features(features)


    rsp = features['RS Power'].values
    # 8.用户建筑物高度m
    # Hm = features['Building Height'].values

    # pfs = np.array([d, h_bs, h_ue,
    #                     h_bs*d, deltaH, betaV])
    pfs = np.array([d, Hm, h_ue,
                    deltaH, deltaHv])
    print("shape:", np.shape(pfs))

    # if is_training:
    #     PL = (PL-61.2) / 90.9
    # else:
    #     PL = (PL-60.2) / 88.9

    X_test = pfs.T

    import pickle
    with open(path, 'rb') as fr:
        model = pickle.load(fr)

        test_RMSE, test_MAPE = 0, 0
        test_cnt = 0
        # precision, accuracy,f1 = 0,0,0
        # for i in range(len(X_test)):
        test_cnt += 1

        rsp = rsp[0]
        print("rsp shape:", np.shape(rsp), rsp)
        # X_test = X_test[:, :-1]
        print("X_test shape:", np.shape(X_test))
        # print(X_test[:10])

        # X_test = X_test[:,:]
        # print("X_test shape:",np.shape(X_test))

        prediction = model.predict(X_test)
        print("PREDICTION shape:", np.shape(prediction))

    return rsp - prediction

def get_deltaHv(features, map_data):
    # 3.垂直下倾角
    betaV = features['Electrical Downtilt'].values[0] + features['Mechanical Downtilt'].values[0]
    # 6.基站塔顶高度m
    Hb = features['Height'].values[0] + features['Cell Building Height'].values[0] + features['Cell Altitude'].values[0]+10

    L = np.zeros([64,64])
    for i in range(64):
        for j in range(64):
            l = np.sqrt(np.square(i-32) + np.square(j-0))
            L[i,j] = l
    L = np.array(L)
            # 16.信号线与用户栅格建筑顶的相对高度m
    print(betaV, Hb)
    deltaHv = Hb - (L * np.tan(np.deg2rad(betaV+5))) - map_data
    return deltaHv

import seaborn as sns
def fig_range(RSRP_map):
    data = RSRP_map[map_mask.astype('bool')]
    # data = np.ndarray.flatten(RSRP_map)
    print(np.shape(data),data)

    plt.figure(figsize=(3.8, 1.5))
    sns.distplot(data)
    plt.xlabel("RSRP range")
    plt.show()

def mape(x1,x2):
    # return (x1-x2)/(x2)
    return (x1 - x2)

def loss_show(pred_RSRP_map, RSRP_map, title):
    true = RSRP_map[map_mask.astype('bool')]
    N = len(pred_RSRP_map)
    ig, ax = plt.subplots(N, 1)
    # preds = []
    for i in range(N):
        if len(pred_RSRP_map[i]) != len(true):
            tmp = mape(pred_RSRP_map[i][map_mask.astype('bool')], true)
        else:
            tmp = mape(pred_RSRP_map[i], true)
        # preds.append(tmp)

        sns.distplot(tmp,ax=ax[i])
        ax[i].set_title(title[i])
    plt.show()



    # data = np.ndarray.flatten(RSRP_map)
    # print(np.shape(data),data)



data_path = "./dataset/csv_set_h50_64/testset/"

test_filenames = cls_ult.get_filename("./dataset/csv_set_h50_64/testset/")

cnt=0
SelectedIndex = 5
for name, context in test_filenames:
    print(cnt, len(test_filenames))
    cnt+=1
    if cnt==SelectedIndex:
        # print(name,context)
        features = pd.read_csv(context)
        map_mask, map_att, map_building, map_deltaHv, map_ci, map_rsrp = get_draw_map(features, 64, 64)
        map_building = pre_data(map_building, map_mask)
        map_att = pre_data(map_att, map_mask,need_filter=False)
        map_data = map_building + map_att


        # deltaHv = get_deltaHv(features, map_data)
        # deltaHv = maxmin_norm(deltaHv,map_mask)

        draw_building_3D(map_data, map_mask, map_rsrp, features)

        # map_rsrp_pred_spm = SPM(features,64,64)

        # draw_inputs_2D(map_data, 'Grand truth on topographic projection', map_rsrp)
        # draw_inputs_2D_clear(deltaHv, 'Topographic projection of deltaHv', map_rsrp)

        # map_rsrp_pred1 = GAN_model(features, "./model/GAN_SC_1/g19")
        # map_rsrp_pred2 = GAN_model(features, "./model/UNET_SC/g55")
        # map_rsrp_pred_gan = GAN_model(features, "./model/GAN_SC_1/g91",fine_turning=True)

        # map_rsrp_pred = GAN_model_org(features, "./model/GAN_SC/g82")

        # map_rsrp_pred = np.reshape(map_rsrp_pred, [64, 64])
        # print(np.shape(map_rsrp_pred), np.shape(map_rsrp))

        # map_residual = pre_data(map_rsrp-map_rsrp_pred_spm, map_mask, need_filter=False)
        # map_rsrp_pred_spm = pre_data(map_rsrp_pred_spm, map_mask)
        # draw_residual_2D('Ground Truth', map_rsrp, map_mask)
        # draw_residual_2D('Ground Truth', map_rsrp_pred_spm, map_mask)

        # draw_residual_2D('Predicted Residual', map_residual, map_mask)
        # draw_inputs_2D_clear('Predicted by SPM', map_rsrp_pred_spm, map_mask)
        # draw_inputs_2D(map_data, 'Prediction results by GAN on topographic projection', map_rsrp_pred_gan)


        # draw_inputs_2D(map_data, 'Prediction results by GAN on topographic projection', map_rsrp_pred)

        # draw_T_F_2D(map_data, 'Prediction results by GAN', map_rsrp_pred_gan, true_rsrp = map_rsrp)

        # ML_pred1 = ML_model(features, path='./model/ML/rfr_sc.pickle')
        # ML_pred2 = ML_model(features, path='./model/ML/knn_sc.pickle')
        # # print("ML_pred shape:", np.shape(ML_pred))
        # draw_inputs_2D(map_data, 'Prediction results by rfr on topographic projection', ML_pred, ML=True)
        #
        # draw_T_F_2D(map_data, 'Prediction results by RFR', ML_pred, true_rsrp=map_rsrp, ML=True)

        # fig_range(map_rsrp-map_rsrp_pred_spm)
        # loss_show([map_rsrp_pred_spm,
        #            ML_pred2,
        #            ML_pred1,
        #            ],
        #           map_rsrp,
        #           ['Loss of SPM',
        #            'Loss of KNN',
        #            'Loss of GAN',
        #            ])
