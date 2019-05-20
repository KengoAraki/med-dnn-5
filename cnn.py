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
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Reshape
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

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
    X /= 255.0
    Y = Y.astype('int32')

    return X, Y


# train用データとtest用データ
x, y = create_dataset(train_images, train_labels)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=111)


# validation用データ
x_valid, y_valid = create_dataset(valid_images, valid_labels)


# Convolutionの出力側のチャンネル数
FIXME_1 = 64
FIXME_2 = 128
FIXME_3 = 128
FIXME_4 = 128
FIXME_5 = 128

print('x_train_shape:{}\nx_test_shape:{}'.format(x_train.shape, x_test.shape))
# %%


def model():
    input_shape = (256, 256, 1)
    # モデルの作成
    model = Sequential()

    # モデルにレイヤーを積み上げる
    model.add(Conv2D(FIXME_1, kernel_size=(5, 5), strides=(
        2, 2), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(FIXME_2, kernel_size=(5, 5), strides=(
        2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(FIXME_3, kernel_size=(3, 3), strides=(
        1, 1), padding='same', activation='relu'))
    model.add(Conv2D(FIXME_4, kernel_size=(3, 3), strides=(
        1, 1), padding='same', activation='relu'))
    model.add(Conv2D(FIXME_5, kernel_size=(1, 1), strides=(
        1, 1), padding='same'))
    # model.add(Conv2DTranspose(1, kernel_size=(32, 32),
    #                           strides=(16, 16)))
    # model.add(Cropping2D(cropping=((8, 8), (8, 8))))
    # model.add(Activation('sigmoid'))
    model.add(Conv2DTranspose(1, kernel_size=(1, 1),
                              strides=(16, 16), activation='sigmoid'))

    # modelの概要
    model.summary()

    return model


model = model()
# %%


def train_model(train_model):
    # 訓練プロセスの定義
    optimizer = Adam(lr=0.00001)
    train_model.compile(optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['binary_accuracy'])

    # 実行
    history = train_model.fit(x_train, y_train, batch_size=128,
                              epochs=200, validation_data=(x_test, y_test), verbose=1)

    # モデルの保存
    train_model.save('fcn_model')

    return history


history = train_model(model)
# %%


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


fcn_model = load_model('fcn_model')
pred_val = predict(x_valid, fcn_model)
print('pred_val:{}'.format(pred_val))

show_predicts(fcn_model, pred_val, y_valid)
# %%
