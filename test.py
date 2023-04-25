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

def pre_data(map_data, map_mask, need_filter=True):
    from scipy.interpolate import griddata
    import scipy.signal as signal
    points = np.nonzero(map_mask)
    values = map_data[map_mask.astype('bool')]  # 已知散点的值
    # print(len(points), len(values))

    xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    # print(xi, yi)

    znew = griddata(points, values, (xi, yi), method='nearest')  # 进行插值

    if need_filter:
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波
        znew = signal.medfilt2d(np.array(znew), kernel_size=3)  # 二维中值滤波

    return znew

def filter_height(map_data):
    import operator
    tmp_list = list(np.reshape(map_data,[np.shape(map_data)[0]*np.shape(map_data)[1]]))

    dict_x = {}
    for item in tmp_list:
        dict_x[item] = tmp_list.count(item)

    sorted_x = sorted(dict_x.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x)

def draw_building_3D(map_data, map_mask,map_rsrp, features):

    col = np.nonzero(map_mask)
    map_rsrp = map_rsrp[map_mask.astype('bool')]

    X = np.arange(0, 64, step=1)  # X轴的坐标
    Y = np.arange(0, 64, step=1)  # Y轴的坐标
    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    bottom = np.ones_like(X)  # 设置柱状图的底端位值
    print(np.min(map_data))
    bottom = bottom * np.min(map_data)

    tmp = map_data
    tmp[32,0] = np.min(map_data)
    Z = tmp-np.min(map_data)
    print(np.shape(tmp))
    Z = Z.ravel()  # 扁平化矩阵
    print(np.shape(Z),Z)
    width = height = 1  # 每一个柱子的长和宽
    colors = plt.cm.coolwarm(Z.flatten()/float(Z.max()))

    # 绘图设置
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴

    pic = ax.bar3d(X, Y, bottom, width, height, Z,
             color= colors,
             shade=True)  # # 坐标轴设置

    xi, yi = np.mgrid[0:63:64j, 0:63:64j]
    cset = ax.contourf(yi, xi, map_data, zdir='z',
                       alpha=0.8,
                       offset=0,
                       # cmap=cm.coolwarm
                       )

    # 设置柱子属性
    H_BS = features['Height'].values[0] \
           + features['Building Height'].max() - 5
           # + features['Cell Building Height'].values[0] \
           # + features['Cell Altitude'].values[0]

    height = 200  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 1  # 柱子的长和宽
    bottom = map_data[32,0]
    X = 32
    Y = 0
    print(bottom,H_BS)
    c = ['green']
    ax.bar3d(X, Y, bottom, width, depth, H_BS,
             color= c, shade=True, alpha=1.0, zorder=4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Height')
    ax.set_zlim3d(0, 600)

    # fig.colorbar(pic, shrink=0.5, aspect=8)

    plt.show()


def draw_inputs_2D(data, title, map_rsrp, ML=False):
    map_rsrp_min = -140
    map_rsrp_max = -70

    # data = np.rot90(data, -1)
    import matplotlib as mpl
    fig,ax = plt.subplots(1, 1, figsize=(8,8))
    # vnorm = mpl.colors.Normalize(vmin=-120, vmax=-70)
    surf = ax.imshow(data,
                     # norm=vnorm,
                     alpha=0.6,
                     # cmap=cm.coolwarm
                     )

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
    # ax.scatter(col[1], col[0],
    #            c=c_list,
    #            s=10,
    #            alpha=1.0,
    #            label='RSRP')


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

data_path = "./dataset/csv_set_h50_64/testset/"

test_filenames = cls_ult.get_filename("./dataset/csv_set_h50_64/testset/")

cnt=0
SelectedIndex = 1
for name, context in test_filenames:
    print(cnt, len(test_filenames))
    cnt+=1
    if cnt==SelectedIndex:
        # print(name,context)
        features = pd.read_csv(context)
        map_mask, map_att, map_building, map_deltaHv, map_ci, map_rsrp = get_draw_map(features, 64, 64)
        map_building = pre_data(map_building, map_mask,need_filter=False)
        # print("log:", np.shape(map_building))
        # filter_height(map_building)
        map_att = pre_data(map_att, map_mask,need_filter=False)

        map_data = map_building + map_att
        # H_BS = features['Height'].values[0]\
        #        + features['Cell Building Height'].values[0]\
        #        + features['Cell Altitude'].values[0]
        # print('H_BS:', H_BS)
        # map_data[63,32] = H_BS

        filter_height(map_data)
        # draw_building_3D(map_data, map_mask, map_rsrp, features)
        draw_inputs_2D(map_data, '', map_rsrp)

        break