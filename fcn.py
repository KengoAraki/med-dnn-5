# %%
from matplotlib import pyplot as plt
import numpy as np
import glob
import copy
import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

# from PIL import Image
# # PILライブラリで画像を読み込む
# img = np.asarray(Image.open(
#     '/home/kengoaraki/study-machine-learning/med-dnn/med-dnn-5/train/image/000.png'))
# label = np.asarray(Image.open(
#     '/home/kengoaraki/study-machine-learning/med-dnn/med-dnn-5/train/label/000.png'))

# # matplotlibライブラリを使って2つの画像を並べて表示
# fig, axes = plt.subplots(1, 2)
# axes[0].set_axis_off()
# axes[0].imshow(img, cmap='gray')
# axes[1].set_axis_off()
# axes[1].imshow(label, cmap='gray')
# plt.show()

train_images = sorted(glob.glob(
    '/home/kengoaraki/study-machine-learning/med-dnn/med-dnn-5/train/image/*.png'))
train_labels = sorted(glob.glob(
    '/home/kengoaraki/study-machine-learning/med-dnn/med-dnn-5/train/label/*.png'))
valid_images = sorted(glob.glob(
    '/home/kengoaraki/study-machine-learning/med-dnn/med-dnn-5/val/image/*.png'))
valid_labels = sorted(glob.glob(
    '/home/kengoaraki/study-machine-learning/med-dnn/med-dnn-5/val/label/*.png'))


def create_dataset(image_files, label_files):
    X = []
    Y = []

    for image in image_files:
        img = img_to_array(load_img(image, color_mode='grayscale'))
        X.append(img)

    for label in label_files:
        img = img_to_array(load_img(label, color_mode='grayscale'))
        Y.append(img)

    # arrayに変換
    X = np.asarray(X)
    Y = np.asarray(Y)
    # 画素値を0から1の範囲に変換
    X = X.astype('float32')
    X = X / 255.0
    Y = Y.astype('int32')
    return X, Y


# train用データとtest用データ
x, y = create_dataset(train_images, train_labels)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=111)

# train用・test用データを(画像数，次元数)にreshape
x_train = x_train.reshape(x_train.shape[0], 256*256)
x_test = x_test.reshape(x_test.shape[0], 256*256)
y_train = y_train.reshape(y_train.shape[0], 256*256)
y_test = y_test.reshape(y_test.shape[0], 256*256)

print('x_train_shape:{}\nx_test_shape:{}'.format(x_train.shape, x_test.shape))
# %%

# validation用データ
x_valid, y_valid = create_dataset(valid_images, valid_labels)

# validation用データを(画像数，次元)にreshape
x_valid = x_valid.reshape(x_valid.shape[0], 256*256)


def model():
    # モデルの作成
    model = Sequential()

    # モデルにレイヤーを積み上げる
    model.add(Dense(100, input_shape=(256*256, ), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(256 * 256, activation='sigmoid'))

    model.summary()

    return model


model = model()
# %%


def train_model(train_model):
    # 訓練プロセスの定義
    optimizer = Adam(lr=0.001)
    train_model.compile(optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['binary_accuracy'])

    # 実行
    history = train_model.fit(x_train, y_train, batch_size=64,
                              epochs=20, validation_data=(x_test, y_test), verbose=1)

    # モデルの保存
    train_model.save('FCL_model')

    return history


history = train_model(model)

# %%


# def run_model():
#     # モデルの作成
#     model = Sequential()

#     # モデルにレイヤーを積み上げる
#     model.add(Dense(100, input_shape=(256*256, ), activation='relu'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(256*256, activation='sigmoid'))

#     # 訓練プロセスの定義
#     optimizer = Adam(lr=0.001)
#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics=['binary_accuracy'])

#     # 実行
#     run_result = model.fit(x_train, y_train, batch_size=64,
#                            epochs=20, validation_data=(x_test, y_test), verbose=1)

#     # モデルの保存
#     model.save('FCL_model')

#     return run_result


# history = run_model()

# accuracyの表示
plt.subplot(121)
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')

# lossの表示
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'], loc='lower right')

plt.show()

# %%


def predict(data, model):

    pred_val = model.predict(data, batch_size=1, verbose=1)
    pred_val = pred_val.reshape(pred_val.shape[0], 256, 256, 1)
    pred_val = pred_val > 0.5
    return pred_val


def show_predicts(model, pred_val, y_val, n_sample=3):

    for i in range(n_sample):
        p_v = copy.copy(pred_val[i])
        y_v = copy.copy(y_val[i])
        fig, axes = plt.subplots(1, 2)

        axes[0].set_axis_off
        axes[0].imshow(array_to_img(p_v), cmap='gray')

        axes[1].set_axis_off
        axes[1].imshow(array_to_img(y_v), cmap='gray')

        plt.show()


# y_valid = y_valid.reshape(y_valid.shape[0], 256 * 256)
# y_valid = y_valid.reshape(y_valid.shape[0], 256, 256, 1)
# print('y_val_shape:{}'.format(y_valid.shape))
fcl_model = load_model('FCL_model')
pred_val = predict(x_valid, fcl_model)
print('pred_val:{}'.format(pred_val))
# print('pred_val_shape:{}\ny_val_shape:{}'.format(pred_val.shape, y_valid.shape))

show_predicts(fcl_model, pred_val, y_valid)
# %%
