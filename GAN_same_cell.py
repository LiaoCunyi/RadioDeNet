import cls_ult
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow import keras
from    matplotlib import pyplot as plt
import matplotlib as mpl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from gd_rsrp import Discriminator, Generator
batch_size = 1
weight = 64
height = 64
input_image_channel = 1



generator = Generator()
generator.build(input_shape=[(batch_size, weight, height, input_image_channel),(batch_size, weight, height, 1)])
# generator.build(input_shape=(batch_size, weight, height, 1))
generator.summary()
discriminator = Discriminator()
discriminator.build(input_shape=[(batch_size, weight, height, 1), (batch_size, weight, height, 1)])
discriminator.summary()

# model_dir = "./model/GAN_SC/g82"
# model_dir = "./model/GAN_SC_1/g91"
# generator.load_weights(model_dir)

# g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
# d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

g_optimizer = keras.optimizers.Adam(learning_rate=2e-3, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=2e-3, beta_1=0.5)


def discriminator_loss(disc_real_output, disc_generated_output):
    # [1, 30, 30, 1] with [1, 30, 30, 1]
    # print(disc_real_output.shape, disc_generated_output.shape)
    real_loss = keras.losses.binary_crossentropy(
                    tf.ones_like(disc_real_output), disc_real_output, from_logits=True)

    generated_loss = keras.losses.binary_crossentropy(
                    tf.zeros_like(disc_generated_output), disc_generated_output, from_logits=True)

    real_loss = tf.reduce_mean(real_loss)
    generated_loss = tf.reduce_mean(generated_loss)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss



def generator_loss(disc_generated_output, gen_output, target):

    LAMBDA = 100

    gan_loss = keras.losses.binary_crossentropy(
                tf.ones_like(disc_generated_output), disc_generated_output, from_logits=True)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    gan_loss = tf.reduce_mean(gan_loss)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

def generate_images(test_input, prediction, tar, epoch):


    fig,ax = plt.subplots(1, 3, figsize=(8*3,8))
    axes = ax.flatten()

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Delta H', 'Ground Truth', 'Predicted Image']

    vnorm1 = mpl.colors.Normalize(vmin=0, vmax=1)
    vnorm2 = mpl.colors.Normalize(vmin=64.45, vmax=134.7)

    ax0 = axes[0].imshow(display_list[0], norm=vnorm1)
    ax1 = axes[1].imshow(display_list[1], norm=vnorm2)
    ax2 = axes[2].imshow(display_list[2], norm=vnorm2)
    # ax1 = axes[1].imshow(display_list[1])
    # ax2 = axes[2].imshow(display_list[2])

    fig.colorbar(ax0, ax=axes[0], fraction=0.05, pad=0.05)
    # fig.colorbar(ax1, ax=axes[1], fraction=0.05, pad=0.05)
    # fig.colorbar(ax2, ax=axes[2], fraction=0.05, pad=0.05)
    fig.colorbar(ax1, ax=[axes[1],axes[2]], fraction=0.05, pad=0.05)

    n = 3
    for i in range(3):
        axes[i].set_title(title[i])


    plt.savefig('images/epoch%d.png'%epoch)
    print('saved images.')
    # plt.show()

def maxmin_norm(data, mask, is_MASK=True):
    if is_MASK:
        data = (data-data[mask.astype('bool')].min()) \
                     / (data[mask.astype('bool')].max()
                        - data[mask.astype('bool')].min())
        data = data * mask
    else:
        data = (data - data.min()) / (data.max() - data.min())
    return data

def z_norm(data, mask):
    data = (data - data[mask.astype('bool')].mean()) \
                 / (data[mask.astype('bool')].std())
    data = data * mask
    return data

def RMSE(A,B):
    return np.sqrt(np.mean(np.power((A - B), 2)))
def MAPE(A,B):
    return np.nanmean(((A - B)/B)) * 100

def caculate_eval_score(cls_true, cls_pred):
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

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    precision = precision_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted')
    return precision,accuracy,f1

def main(dataset,testset):
    print(np.shape(dataset))
    epochs = 500
    # y_pred, map_data, map_rsrp, map_mask = dataset
    # 以元组方式 生成Dataset数据集
    dataset_tuple = tf.data.Dataset.from_tensor_slices(dataset)

    best_score = 10
    for epoch in range(epochs):
        start = time.time()
        # db_tuple = dataset_tuple.batch(batch_size)
        db_tuple = dataset_tuple.shuffle(400).batch(batch_size)


        for step, inputs in enumerate(db_tuple):

            ML_pred = np.reshape(inputs[:, 0, :, :],[batch_size, weight, height, 1])
            EM = np.reshape(inputs[:, 1, :, :],[batch_size, weight, height, 1])

            # input_image = np.reshape(inputs[:, 0:2, :, :],[batch_size, weight, height, 2])
            ground_true = np.reshape(inputs[:, 2, :, :], [batch_size, weight, height, 1])
            mask = np.reshape(inputs[:, 3, :, :], [batch_size, weight, height, 1])

            # print(step, '-->', "elment shape:", inputs.shape)
            # input_image = np.reshape(inputs[:, 0, :, :],[batch_size, weight, height, 1])
            # in_mask = np.reshape(inputs[:, 1, :, :],[batch_size, weight, height, 1])
            # ground_ture = np.reshape(inputs[:, 2, :, :],[batch_size, weight, height, 1])
            # fitting_pred = np.reshape(inputs[:, 3, :, :],[batch_size, weight, height, 1])
            # out_mask = np.reshape(inputs[:, 4, :, :],[batch_size, weight, height, 1])

            # input_image = np.transpose(input_image, [0,2,3,1])

            ground_true = maxmin_norm(ground_true, mask, is_MASK=True)
            ML_pred = maxmin_norm(ML_pred, mask, is_MASK=False)
            EM = maxmin_norm(EM, mask, is_MASK=False)

            input_image = np.reshape(np.array([ML_pred,EM]), [batch_size, weight, height, 2])


            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # get generated pixel2pixel image
                gen_output = generator([ML_pred ,ground_true], training=True)

                gen_output = gen_output * mask

                # fed real pixel2pixel image together with original image
                disc_real_output = discriminator([EM, ground_true], training=True)
                # fed generated/fake pixel2pixel image together with original image
                disc_generated_output = discriminator([EM, gen_output], training=True)

                gen_loss = generator_loss(disc_generated_output, gen_output, ground_true)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            # generator_gradients = [tf.clip_by_norm(g, 15) for g in generator_gradients]
            g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

            discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            # discriminator_gradients = [tf.clip_by_norm(g, 15) for g in discriminator_gradients]
            d_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            if step% 20 == 0:
                # print(disc_loss.shape, gen_loss.shape)
                print(epoch, step, float(disc_loss), float(gen_loss), RMSE(gen_output, ground_true))
                # generate_images([input_deltaH], prediction, target, epoch)



        if epoch % 1 == 0:
            test_T = time.time()
            test_RMSE = []
            test_MAPE = []
            precision, accuracy, f1 = [], [], []
            test_cnt = 0
            inputs = testset
            bs = len(testset)
            ML_pred = np.reshape(inputs[:, 0, :, :], [bs, weight, height, 1])
            EM = np.reshape(inputs[:, 1, :, :], [bs, weight, height, 1])

            input_image = np.reshape(inputs[:, 0:2, :, :], [bs, weight, height, 2])
            ground_true = np.reshape(inputs[:, 2, :, :], [bs, weight, height, 1])
            mask = np.reshape(inputs[:, 3, :, :], [bs, weight, height, 1])

            ML_pred = maxmin_norm(ML_pred, mask, is_MASK=False)
            EM = maxmin_norm(EM, mask, is_MASK=False)

            input_image = np.reshape(np.array([ML_pred, EM]), [bs, weight, height, 2])

            prediction = generator([ML_pred ,ground_true], training=False)
            prediction = prediction * (ground_true[mask.astype('bool')].max()
                                       - ground_true[mask.astype('bool')].min()) \
                                        + ground_true[mask.astype('bool')].min()
            # prediction = prediction * target[mask.astype('bool')].std() + target[mask.astype('bool')].mean()
            prediction = prediction

            t_ture = ground_true[mask.astype('bool')]
            t_pred = prediction[mask.astype('bool')]

            test_RMSE.append(RMSE(t_pred, t_ture))
            test_MAPE.append(MAPE(t_pred, t_ture))


            test_cnt += 1

            test_rmse = np.mean(test_RMSE)
            test_mape = np.mean(test_MAPE)

            print('===Time taken for epoch {} is {} sec.\n'
                  '===The test RMSE is {}. The test MAPE is {}\n'
                  '===Time for predicting testdataset is {}'.format(
                epoch + 1, time.time() - start, test_rmse, test_mape, time.time() - test_T
            ))
            # print("Precision:", np.mean(precision), "Acc:", np.mean(accuracy), "F1_score:", np.mean(f1))
            if np.mean(test_rmse) < best_score:
                best_score = np.mean(test_rmse)
                # tf.saved_model.save(generator,'./model/GAN/generator')
                generator.save_weights('./model/GAN/GAN')
                print('--------------------------------------------------')
                print("--------BEST SCORE:",best_score,"-----------")
                print('--------------------------------------------------')



def get_dataset(filenames, is_training = True, div_num = 0.8):

    dataset = []
    cnt = 0
    RSP = []
    for name,context in filenames:
        cnt+=1
        # print(name,context)
        features = pd.read_csv(context)

        div_len = np.int(len(features) * div_num)
        train_data = features.loc[0:div_len - 1]
        test_data = features.loc[div_len:len(features)]
        # print(len(train_data), len(test_data))

        map_deltaH, input_mask, map_pl, map_pl_pred, building_mask = get_map_features(features, weight, height)

        train_output_mask = get_output_mask(train_data, weight, height)

        test_output_mask = get_output_mask(test_data, weight, height)

        RSP.append(features['RS Power'].values[0])

        # #normlization
        map_deltaH = maxmin_norm(map_deltaH, input_mask)

        # print("map_deltaH:", np.shape(map_deltaH))
        # print("input_mask:", np.shape(input_mask))
        # print("map_pl:", np.shape(map_pl))
        # print("map_pl_pred:", np.shape(map_pl_pred))
        # print("output_mask:", np.shape(output_mask))


        # map_deltaHv = maxmin_norm(map_deltaHv, map_mask)
        # map_deltaH = z_norm(map_deltaH, map_mask)
        # map_deltaHv = z_norm(map_deltaHv, map_mask)

        # if is_training:
        #     map_rsrp = maxmin_norm(map_rsrp, map_mask)

        # map_rsrp = maxmin_norm(map_rsrp, map_mask)

        dataset.append([map_deltaH, input_mask, map_pl, map_pl_pred, train_output_mask, test_output_mask])

        if cnt == 1:
            break
    print("dataset:", np.shape(dataset))
    return np.array(dataset),RSP

# train_data = get_dataset(train_filenames)
# val_data = get_dataset(train_filenames, is_training = False)
# test_data = get_dataset(test_filenames, is_training = False)
#
# main(train_data, val_data, test_data)

print("Read dataset...")
# from Features2img import get_map_features, get_output_mask

train_filenames = cls_ult.get_filename("./dataset/csv_set_h50_64/testset/")
# test_filenames = cls_ult.get_filename("./dataset/csv_set_h50/testset/")

train_dataset = np.load('./dataset/train_dataset.npy')
test_dataset = np.load('./dataset/test_dataset.npy')
print("dataset NAN?:", np.isnan(train_dataset).any())

main(train_dataset,test_dataset)




