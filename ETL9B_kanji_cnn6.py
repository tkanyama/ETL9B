import numpy as np
import cv2, pickle
from sklearn.model_selection import train_test_split
# import plaidml.keras
# plaidml.keras.install_backend()
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt
from os.path import expanduser
# home = expanduser("~")
home = 'E:/DATA'
import time
t0 = time.time()

# データファイルと画像サイズの指定
data_file1 = home + "/ETL/ETL9B/ETL9B_32.pickle"
jis_code_file1 = home + '/ETL/ETL9B/ETL9BJISCODE.picle'
code_file = home + '/ETL/ETL9B/ETL9B_KANA_JIS_CODE.picle'
data_file2 = home + "/ETL/ETL1/ETL1/katakana_32.pickle"
jis_code_file2 = home + '/ETL/ETL1/ETL1/KANACODE.picle'
# im_size = 25
# out_size = 46 # ア-ンまでの文字の数

# カタカナ画像のデータセットを読み込む --- (*1)
# (im_size, out_size, code_mat, data) = pickle.load(open(data_file, "rb"))
(im_size1, out_size1, data1) = pickle.load(open(data_file1, "rb"))
(im_size2, out_size2, data2) = pickle.load(open(data_file2, "rb"))
n = len(data2)
for i in range(n):
    data2[i][0] =data2[i][0] + out_size1
data = np.concatenate([data1, data2], 0)
data = data.tolist()
(code_mat1) = pickle.load(open(jis_code_file1, "rb"))
(code_mat2) = pickle.load(open(jis_code_file2, "rb"))
code_mat = np.concatenate([code_mat1, code_mat2], 0)
code_mat = code_mat.tolist()

pickle.dump((code_mat), open(code_file, "wb"))

im_color = 1 # 画像の色空間/グレイスケール
im_size = im_size1
out_size = out_size1 + out_size2
in_shape = (im_size, im_size, im_color)

# data  = pickle.load(open(data_file, "rb"))
# 画像データを変形して0-1の範囲に直す --- (*2)
y = []
x = []
for d in data:
    (num, img) = d
    img = img.astype('float').reshape(
      im_size, im_size, im_color) / 255
    y.append(keras.utils.np_utils.to_categorical(num, out_size))
    x.append(img)
x = np.array(x)
y = np.array(y)

# 学習用とテスト用に分離する
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# datagen = ImageDataGenerator(rotation_range=15,
#                              shear_range=0.15,
#                              zoom_range=[1.0, 1.20])
datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.15,
        zoom_range=[1.0, 1.40],
        horizontal_flip=False,
        fill_mode='constant',
        cval=0)

datagen.fit(x_train)


# CNNモデル構造を定義 --- (*3)

model = Sequential()

# model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))


model.add(Conv2D(32,
          kernel_size=(3, 3),
          activation='relu',
          input_shape=in_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(out_size, activation='softmax'))

model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
bsize = 2048
epoch_n = 150

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=bsize), samples_per_epoch=x_train.shape[0],
                    nb_epoch=epoch_n, validation_data=(x_test, y_test))
# 学習を実行して評価 --- (*4)
# hist = model.fit(x_train, y_train,
#           batch_size=128,
#           epochs=12,
#           verbose=1,a
#           validation_data=(x_test, y_test))
# モデルを評価
model.save(home + '/ETL/ETL9B/etl9b_model6-1_{}_{}_{}_01.h5'.format(im_size, bsize, epoch_n))
# model.save_weights(home + '/ETL/ETL9B/etl9b_{}_weight3_01.h5'.format(im_size))
score = model.evaluate(x_test, y_test, verbose=1)
print('正解率=', score[1], 'loss=', score[0])
print('time={}sec'.format(time.time() - t0))

# 学習の様子をグラフへ描画 --- (*5)
# 正解率の推移をプロット
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ロスの推移をプロット
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

