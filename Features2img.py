# import model_ult, cls_ult
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import scipy.signal as signal

def pre_data(map_data, map_mask, need_filter=True):
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

def get_map_features(features,w,h):
    pb_data = features

    # f = features['Frequency Band'].values
    #
    # h_bs = features['Cell Altitude'].values + \
    #        features['Cell Building Height'].values + \
    #        features['Height'].values - att_min
    # h_ue = features['Altitude'].values \
    #        - att_min + 1.0
    #
    # hm = features['Building Height'].values

    # h_bs = features['Cell Building Height'].values +\
    #        features['Height'].values
    # h_ue = features['Altitude'].values - att_min + 1.0


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

    # 12.用户地物索引-
    UCI = pb_data['Clutter Index']

    # 13.水平余弦相似度A
    n1 = np.array([x, y])
    # n2 = np.array([np.tan(pb_data['Azimuth']*np.pi/180), 1-pb_data['Y']+pb_data['Y']])
    n2 = np.array([np.sin(pb_data['Azimuth'] * np.pi / 180), np.cos(pb_data['Azimuth'] * np.pi / 180)])
    num = np.sum(np.multiply(n1, n2), axis=0)  # 若为行向量则 A * B.T

    denom = np.sqrt(np.sum(np.square(n1), axis=0)) * np.sqrt(np.sum(np.square(n2), axis=0))
    cosA = num / denom  # 余弦值''

    # 14.垂直倾角B°
    B = np.rad2deg(np.arctan(np.divide((deltaH), L * cosA))) - betaV
    cosB = np.cos(np.deg2rad(B))

    # 15.THETA(C)°
    v1 = np.array([L * np.sin(np.arccos(cosA)), L * cosA, -deltaH])
    # v2 = np.array([L-L,deltaH/np.tan(np.deg2rad(betaV)),-deltaH])
    v2 = np.array([L - L, np.cos(np.deg2rad(betaV)), -np.sin(np.deg2rad(betaV))])
    num = np.sum(np.multiply(v1, v2), axis=0)
    denom = np.sqrt(np.sum(np.square(v1), axis=0)) * np.sqrt(np.sum(np.square(v2), axis=0))
    cosC = num / denom  # 余弦值

    # 16.信号线与用户栅格建筑顶的相对高度m
    deltaHv = Hb - (L * np.tan(np.deg2rad(betaV))) - Husr - Hm

    # # 9.有效高度m
    # deltaH = Hb - Husr

    RSRP =pb_data['RSRP'].values

    PL = pb_data['RS Power'].values - RSRP

    # fitting model predict
    att_min = features['Altitude'].min()
    f = features['Frequency Band'].values
    h_bs = features['Cell Altitude'].values + \
           features['Cell Building Height'].values + \
           features['Height'].values - att_min
    h_ue = features['Altitude'].values \
           - att_min + 1.0
    ci = features['Clutter Index'].values

    # x = features['X'] - features['Cell X']
    # y = features['Y'] - features['Cell Y']
    # d = np.sqrt(np.square(x) + np.square(y)).values / 1000
    # lgd, lght, lght * lgd, hr, ci

    # K1, K2, K3, K4, K5, K6 = 25.46, 36.24, 17.57, 0, -9.9, 0

    # lght = np.log10(h_bs)
    # lgd = np.log10(d)
    # lgf = np.log10(f)
    # hr = h_ue
    # w1,w2,w3,w4,w5,b = 30.33,25.15,-12.81,-0.016,0.087,35.86
    # w1,w2,w3,w4,w5,b = Ks[0],Ks[1],Ks[2],Ks[3],Ks[4],Ks[5]

    # [[3.0330097e+01  2.5150488e+01 - 1.2812478e+01 - 1.5739869e-02
    #   8.7482892e-02  3.5861874e+01]]
    # [[31.76678     27.625462 - 14.536443 - 0.47451213 - 0.05490435
    #   37.257187]]
    K1, K2, K3, K4, K5, K6 = 25.46, 36.24, 17.57, 0, -9.9, 0

    # f, d, h_bs, h_ue = get_cost_features(features)
    # d = d * 1000
    pl_pred = []
    for i in range(len(f)):
        # pl = K1+K2*np.log10(d[i])+K3*np.log10(h_bs[i])+K4*Diffraction+\
        # K5*np.log10(h_bs[i])*np.log10(d[i])+K6*h_ue[i]+K_clutter
        # pl = K1 + K2 * np.log10(d[i]) + K3 * np.log10(h_bs[i]) + K5 * np.log10(h_bs[i]) * np.log10(d[i]) + K6 * h_ue[i]
        pl = K1 + K2 * np.log10(d[i]) + K3 * np.log10(h_bs[i]) + K5 * np.log10(h_bs[i]) * np.log10(d[i]) + K6 * h_ue[i]

        pl_pred.append(pl)
    # pl_pred = w1 * lgd + w2 * lght + w3 * lght * lgd + w4 * hr + w5 * ci + b
    # print("pl：", pl[0])
    # RSP = features['RS Power'].values
    # RSRP = features['RSRP'].values

    pl_pred = np.array(pl_pred)
    # end fitting model

    x_ = np.array(pb_data['x_'].values).astype(np.int) + w/2
    y_ = np.array(pb_data['y_'].values).astype(np.int)
    # x_ = np.floor(pb_data['x_'].values + w/2, 0)
    # y_ = np.floor(pb_data['y_'].values, 0)


    # print(len(x_), len(y_))

    map_deltaH = np.zeros([w, h])
    map_mask = np.zeros([w, h])
    map_pl = np.zeros([w, h])
    map_building = np.zeros([w, h])
    map_pl_pred = np.zeros([w, h])
    # map_RSP = np.zeros([w, h])
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


                map_pl[xx, yy] = PL[i]
                map_hm[xx, yy] = Hm[i]
                map_att[xx, yy] = Husr[i]

                map_pl_pred[xx, yy] = pl_pred[i]
                # map_RSP[xx, yy] = RSP[i]

                map_deltaH[xx, yy] = deltaHv[i]

                if Hm[i] != 0:
                    map_building[xx, yy] = 1

    # map_hm = pre_data(map_hm, map_mask)
    # map_att = pre_data(map_att, map_mask, need_filter=False)
    # map_deltaH = Hb[0]-map_hm-map_att

    # print("map_deltaH:", map_deltaH, "map_mask:", map_mask)

    return map_deltaH, map_mask, map_pl, map_pl_pred, map_building

def get_map_features_fit(features,w,h, Ks):
    pb_data = features

    # f = features['Frequency Band'].values
    #
    # h_bs = features['Cell Altitude'].values + \
    #        features['Cell Building Height'].values + \
    #        features['Height'].values - att_min
    # h_ue = features['Altitude'].values \
    #        - att_min + 1.0
    #
    # hm = features['Building Height'].values

    # h_bs = features['Cell Building Height'].values +\
    #        features['Height'].values
    # h_ue = features['Altitude'].values - att_min + 1.0


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

    # 12.用户地物索引-
    UCI = pb_data['Clutter Index']

    # 13.水平余弦相似度A
    n1 = np.array([x, y])
    # n2 = np.array([np.tan(pb_data['Azimuth']*np.pi/180), 1-pb_data['Y']+pb_data['Y']])
    n2 = np.array([np.sin(pb_data['Azimuth'] * np.pi / 180), np.cos(pb_data['Azimuth'] * np.pi / 180)])
    num = np.sum(np.multiply(n1, n2), axis=0)  # 若为行向量则 A * B.T

    denom = np.sqrt(np.sum(np.square(n1), axis=0)) * np.sqrt(np.sum(np.square(n2), axis=0))
    cosA = num / denom  # 余弦值''

    # 14.垂直倾角B°
    B = np.rad2deg(np.arctan(np.divide((deltaH), L * cosA))) - betaV
    cosB = np.cos(np.deg2rad(B))

    # 15.THETA(C)°
    v1 = np.array([L * np.sin(np.arccos(cosA)), L * cosA, -deltaH])
    # v2 = np.array([L-L,deltaH/np.tan(np.deg2rad(betaV)),-deltaH])
    v2 = np.array([L - L, np.cos(np.deg2rad(betaV)), -np.sin(np.deg2rad(betaV))])
    num = np.sum(np.multiply(v1, v2), axis=0)
    denom = np.sqrt(np.sum(np.square(v1), axis=0)) * np.sqrt(np.sum(np.square(v2), axis=0))
    cosC = num / denom  # 余弦值

    # 16.信号线与用户栅格建筑顶的相对高度m
    deltaHv = Hb - (L * np.tan(np.deg2rad(betaV))) - Husr - Hm

    # # 9.有效高度m
    # deltaH = Hb - Husr

    RSRP =pb_data['RSRP'].values

    PL = pb_data['RS Power'].values - RSRP

    # fitting model predict
    att_min = features['Altitude'].min()
    f = features['Frequency Band'].values
    h_bs = features['Cell Altitude'].values + \
           features['Cell Building Height'].values + \
           features['Height'].values - att_min
    h_ue = features['Altitude'].values \
           - att_min + 1.0
    ci = features['Clutter Index'].values

    # x = features['X'] - features['Cell X']
    # y = features['Y'] - features['Cell Y']
    # d = np.sqrt(np.square(x) + np.square(y)).values / 1000


    # K2 = Ks[0]
    # K3 = Ks[1]
    # K5 = Ks[2]
    # K6 = Ks[3]
    # K_clutter = Ks[4]
    # b = Ks[5]

    lght = np.log10(h_bs)
    lgd = np.log10(d)
    lgf = np.log10(f)
    hr = h_ue
    # K1, K2, K3, K4, K5, K6 = 25.46, 36.24, 17.57, 0, -9.9, 0



    # print(K2,K2, K3, K5, K6, K_clutter)

    # f, d, h_bs, h_ue = get_cost_features(features)
    # lgd, lght, lght * lgd, hr, ci
    # d = d * 1000
    pl_pred = []
    # for i in range(len(f)):
        # PL = K1+K2*np.log10(d[i])+K3*np.log10(h_bs[i])+K4*Diffraction+\
        # K5*np.log10(h_bs[i])*np.log10(d[i])+K6*h_ue[i]+K_clutter

        # pl = b + K2 * np.log10(d[i]) + K3 * np.log10(h_bs[i]) + K5 * np.log10(h_bs[i]) * np.log10(d[i]) + K6 * h_ue[i] + K_clutter * ci[i]
    pl_pred = Ks[0] * lgd + Ks[1] * lght + Ks[2]* lght * lgd + Ks[3]* hr + Ks[4] * ci + Ks[5]

    # pl_pred.append(pl)
    # print("pl：", pl[0])
    # RSP = features['RS Power'].values
    # RSRP = features['RSRP'].values

    pl_pred = np.array(pl_pred)
    # end fitting model

    x_ = np.array(pb_data['x_'].values).astype(np.int) + w/2
    y_ = np.array(pb_data['y_'].values).astype(np.int)
    # x_ = np.floor(pb_data['x_'].values + w/2, 0)
    # y_ = np.floor(pb_data['y_'].values, 0)


    # print(len(x_), len(y_))

    map_deltaH = np.zeros([w, h])
    map_mask = np.zeros([w, h])
    map_pl = np.zeros([w, h])
    map_building = np.zeros([w, h])
    map_pl_pred = np.zeros([w, h])
    map_RSP = np.zeros([w, h])
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


                map_pl[xx, yy] = PL[i]
                map_deltaH[xx, yy] = deltaHv[i]
                # map_deltaHv[xx, yy] = deltaHv[i]
                map_pl_pred[xx, yy] = pl_pred[i]
                map_RSP[xx, yy] = RSP[i]

                if Hm[i] != 0:
                    map_building[xx, yy] = 1


    # print("map_deltaH:", map_deltaH, "map_mask:", map_mask)

    return map_deltaH, map_mask, map_pl, map_pl_pred, map_building, map_RSP

def get_output_mask(features,w,h):
    x_ = np.array(features['x_'].values).astype(np.int) + w / 2
    y_ = np.array(features['y_'].values).astype(np.int)
    map_mask = np.zeros([w, h])
    for i in range(len(x_)):
        if x_[i] < w:
            if y_[i] < h:
                xx = int(x_[i])
                yy = int(y_[i])
                map_mask[xx, yy] = 1
    return map_mask

def pre_pic(pic):
    pic = 255 * pic
    return pic

def img_inpainting(features):
    batch_size = 1
    weight, height = 64, 64
    input_image = np.reshape(features[0, :, :], [weight, height, 1])
    in_mask = np.reshape(features[1, :, :], [weight, height, 1])
    # ground_ture = np.reshape(inputs[:, 2, :, :], [batch_size, weight, height, 1])
    # fitting_pred = np.reshape(inputs[:, 3, :, :], [batch_size, weight, height, 1])
    # out_mask = np.reshape(inputs[:, 4, :, :], [batch_size, weight, height, 1])
    # print(np.min(input_image))
    import cv2
    from matplotlib import pyplot as plt
    input_image = pre_pic(input_image)
    img_map = cv2.merge([input_image.astype(np.uint8)])
    in_mask = cv2.merge([in_mask.astype(np.uint8)])
    dst_TELEA = cv2.inpaint(img_map, in_mask, 64, cv2.INPAINT_TELEA)
    dst_NS = cv2.inpaint(img_map, in_mask, 64, cv2.INPAINT_NS)

    plt.subplot(221), plt.imshow(img_map)
    plt.title('degraded image')
    plt.subplot(222), plt.imshow(in_mask, 'gray')
    plt.title('mask image')
    plt.subplot(223), plt.imshow(dst_TELEA)
    plt.title('TELEA')
    plt.subplot(224), plt.imshow(dst_NS)
    plt.title('NS')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # inputs = np.load("./dataset/Set_hd50_82.npy")
    # # print(step, '-->', "elment shape:", inputs.shape)
    #
    # for inp in inputs:
    #     img_inpainting(inp)
    #     break

    dataset_path = "./dataset/csv_set_h50_64/trainset/1031201.csv"
    features = pd.read_csv(dataset_path)
    print(features.head())

    map_deltaH, map_mask, map_pl, map_pl_pred, map_building = get_map_features(features, 64, 64)
    # map_pl_pred = features['RSRP'].values[0] - map_pl_pred[map_mask.astype('bool')]
    print("PL:", map_pl.min(), map_pl.max())



    #
    # import cv2
    #
    # map = pre_pic(map_deltaH)
    # img_map_deltaH = cv2.merge([map])
    #
    # map_mask = pre_pic(map_mask)
    # img_map_mask = cv2.merge([map_mask])
    #
    # map_rsrp = pre_pic(map_rsrp)
    # img_map_rsrp = cv2.merge([map_rsrp])
    #
    #
    # cv2.imwrite("./result/img_map_deltaH.png",img_map_deltaH)
    # cv2.imwrite("./result/img_map_mask.png",img_map_mask)
    # cv2.imwrite("./result/img_map_rsrp.png",img_map_rsrp)
