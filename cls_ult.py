import os, time, pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore")


def get_filename(root_dir, debug=False):
    filenames = []
    sample_cnt = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            sample_cnt += 1
            # print("files: ",os.path.join(root, name),name)
            file_name = name
            file_content = os.path.join(root, name)
            filenames.append([file_name,file_content])
            # if debug:
            #     if sample_cnt == 1:
            #         break
    return filenames

def load_dataset_pickle(dataset_path, debug=False):
    t = time.time()
    sample_cnt = 0
    path = get_filename(dataset_path, debug)

    all_data = pd.DataFrame(columns=["Cell Index","Cell X","Cell Y",
                            "Height","Azimuth","Electrical Downtilt",
                          "Mechanical Downtilt","Frequency Band",
                        "RS Power","Cell Altitude",
                          "Cell Building Height","Cell Clutter Index",
                            "X","Y","Altitude","Building Height",
                              "Clutter Index","RSRP"])

    for file_name, file_content in path:
        sample_cnt += 1
        with open(file_content, 'rb') as test:
            pb_data = pickle.load(test)

        # print(pb_data.head())
        all_data = all_data.append(pb_data)

        # if sample_cnt==5:
        #     break

        if debug:
            if sample_cnt==5:
                return pb_data

    print("dataset loaded in %.1f s" % (time.time() - t))
    return all_data

def rotate(pb_data):
    x = pb_data['X'] - pb_data['Cell X']
    y = pb_data['Y'] - pb_data['Cell Y']

    x = x / 5
    y = y / 5

    # A = np.deg2rad(pb_data['Azimuth'])
    A = np.radians(pb_data['Azimuth'])
    x_ = x * np.cos(A) - y * np.sin(A)
    y_ = x * np.sin(A) + y * np.cos(A)

    pb_data['x_'] = x_
    pb_data['y_'] = y_
    pb_data = pb_data.iloc[list(pb_data["x_"] > -40)]
    pb_data = pb_data.iloc[list(pb_data["x_"] < 40)]
    pb_data = pb_data.iloc[list(pb_data["y_"] > 0)]
    pb_data = pb_data.iloc[list(pb_data["y_"] < 80)]
    return pb_data

def rotate_dataset(pb_data, limit_distance=40):
    x = pb_data['X'] - pb_data['Cell X']
    y = pb_data['Y'] - pb_data['Cell Y']

    x = x / 5
    y = y / 5

    # A = np.deg2rad(pb_data['Azimuth'])
    A = np.radians(pb_data['Azimuth'])
    x_ = x * np.cos(A) - y * np.sin(A)
    y_ = x * np.sin(A) + y * np.cos(A)

    pb_data['x_'] = x_
    pb_data['y_'] = y_
    pb_data = pb_data.iloc[list(pb_data["x_"] > -limit_distance/2)]
    pb_data = pb_data.iloc[list(pb_data["x_"] < limit_distance/2)]
    pb_data = pb_data.iloc[list(pb_data["y_"] > 0)]
    pb_data = pb_data.iloc[list(pb_data["y_"] < limit_distance)]
    return pb_data

def get_PFs(pb_data, is_rotate = False, org = False):
    err_flag = 0

    # if pb_data['Electrical Downtilt'][0] + pb_data['Mechanical Downtilt'][0] < 1:
    #     print("Downtilt small! Drop!")
    #     err_flag = 1
    #     pd_Features = []
    #     return pd_Features, err_flag
    # elif pb_data['Height'][0]+pb_data['Cell Altitude'][0] == 0:
    #     print("Cell Altitude is 0! Drop!")
    #     err_flag = 1
    #     pd_Features = []
    #     return pd_Features, err_flag

    if is_rotate:
        pb_data = rotate(pb_data)

    x = pb_data['X'] - pb_data['Cell X']
    y = pb_data['Y'] - pb_data['Cell Y']

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
    deltaH = Hb - Husr
    deltaH_ = Hb - Husr - Hm

    # 10.水平距离m
    L = np.sqrt(np.square(x) + np.square(y))

    # 11.链路距离m
    d = np.sqrt(np.square(L) + np.square(deltaH))

    # 12.用户地物索引-
    UCI = pb_data['Clutter Index']

    #13.水平余弦相似度A
    n1 = np.array([x, y])
    # n2 = np.array([np.tan(pb_data['Azimuth']*np.pi/180), 1-pb_data['Y']+pb_data['Y']])
    n2 = np.array([np.sin(pb_data['Azimuth'] * np.pi / 180), np.cos(pb_data['Azimuth'] * np.pi / 180)])
    num = np.sum(np.multiply(n1, n2),axis=0)  # 若为行向量则 A * B.T

    denom = np.sqrt(np.sum(np.square(n1), axis=0)) * np.sqrt(np.sum(np.square(n2), axis=0))
    cosA = num / denom  # 余弦值''

    # 14.垂直倾角B°
    B = np.rad2deg(np.arctan(np.divide((deltaH), L*cosA))) - betaV
    cosB = np.cos(np.deg2rad(B))

    # 15.THETA(C)°
    v1 = np.array([L * np.sin(np.arccos(cosA)), L*cosA, -deltaH])
    # v2 = np.array([L-L,deltaH/np.tan(np.deg2rad(betaV)),-deltaH])
    v2 = np.array([L - L, np.cos(np.deg2rad(betaV)), -np.sin(np.deg2rad(betaV))])
    num = np.sum(np.multiply(v1, v2), axis=0)
    denom = np.sqrt(np.sum(np.square(v1), axis=0)) * np.sqrt(np.sum(np.square(v2), axis=0))
    cosC = num / denom  # 余弦值

    # 16.信号线与用户栅格建筑顶的相对高度m
    deltaHv = Hb - (L*cosA * np.tan(np.deg2rad(betaV))) - Husr

    # RSRP
    RSRP = pb_data['RSRP']

    if is_rotate:
        x = pb_data['x_']
        y = pb_data['y_']


    pd_DICT = {'CI': CI.values, 'FB': FB.values, 'RSP': RSP.values, 'RSRP': RSRP.values,
               'betaV': betaV.values, 'CCI': CCI.values, 'Hb': Hb.values, 'Husr': Husr.values, 'Hm': Hm.values,
               'deltaH': deltaH_.values, 'deltaHv': deltaHv.values, 'L': L.values, 'D': d.values,
               'UCI': UCI.values, 'cosA': cosA, 'cosB': cosB.values, 'cosC': cosC, 'x':x, 'y':y}
    pd_Features = pd.DataFrame(pd_DICT)
    return pd_Features, err_flag


def sin(x):
    return np.sin(np.radians(x))
def cos(x):
    return np.cos(np.radians(x))
def tan(x):
    return np.tan(np.radians(x))
def get_features(Azi, Cell_X, Cell_Y, X,Y ):
    # 判断方向角确定信号线方程
    if Azi < 90:
        k = tan(90 - Azi)
        b = Cell_Y - k * Cell_X
        x2 = Cell_X + 60
        y2 = k * x2 + b
    elif Azi > 90 and Azi < 180:
        k = tan(270 - Azi)
        b = Cell_Y - k * Cell_X
        x2 = Cell_X + 60
        y2 = k * x2 + b
    elif Azi > 180 and Azi < 270:
        k = tan(270 - Azi)
        b = Cell_Y - k * Cell_X
        x2 = Cell_X - 60
        y2 = k * x2 + b
    elif Azi > 270:
        k = tan(450 - Azi)
        b = Cell_Y - k * Cell_X
        x2 = Cell_X - 60
        y2 = k * x2 + b
    elif Azi == 0:
        x2 = Cell_X
        y2 = Cell_Y + 60
    elif Azi == 90:
        x2 = Cell_X + 60
        y2 = Cell_Y
    elif Azi == 180:
        x2 = Cell_X
        y2 = Cell_Y - 60
    elif Azi == 270:
        x2 = Cell_X - 60
        y2 = Cell_Y

    # 坐标系转换--平移
    x = (X - Cell_X)
    y = (Y - Cell_Y)
    x1_ = x2 - (Cell_X)
    y1_ = y2 - (Cell_Y)

    # 坐标系转换--旋转`
    x_ = x * cos(Azi) - y * sin(Azi)
    y_ = x * sin(Azi) + y * cos(Azi)
    x2_ = (x2 - Cell_X)/5 * cos(Azi) - (y2 - Cell_Y)/5 * sin(Azi)
    y2_ = (x2 - Cell_X)/5 * sin(Azi) + (y2 - Cell_Y)/5 * cos(Azi)
    print(x2, y2)
    print(x2_, y2_)

    line1 = [(0, 0), (x1_ * 3, y1_ * 3)]
    line2 = [(0, 0), (x2_ * 3, y2_ * 3)]
    # (line1_xs, line1_ys) = zip(*line1)

    # return zip(*line1), x, y
    return zip(*line2), x_, y_