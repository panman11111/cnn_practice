from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import optimizers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, random

# 設定値
batch_size = 32
num_classes = 10
epochs = 5
img_rows, img_cols = 28, 28
hidden_units = 50
learning_rate = 1e-6
clip_norm = 1.0
row_hidden = 128
col_hidden = 128

# 乱数シード
seed=0
os.environ['PYTHONHASHSEED'] = '0'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Kerasの手書き文字データセットを使用
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 中身を確認
plt.imshow(x_train[0], cmap="gray_r")

#データフレームに格納
df_y_train=pd.DataFrame(y_train, columns=['number'])

#各数字の出現数をカウントする
df_y_train['number'].value_counts()

#グラフ化する
# sns.countplot(x='number', data=df_y_train) # 学習データに0-9が満遍なく含まれていることを確認

# 次元変換 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# CNNの引数の形式（4次元の形式）に変換
input_shape = (img_rows, img_cols, 1)

# 正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 数値ラベルをワンホットエンコーディング
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# シーケンシャルモデルを作成
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# モデルの概要
# model.summary()

# モデルの可視化
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

# モデルをコンパイル
# optimizer=optimizers.RMSprop(learning_rate=1e-4)
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=optimizer,
#               metrics=['accuracy'])

# 学習
# result = model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# 評価
# score = model.evaluate(x_test, y_test, verbose=0)
# print('CNN test loss   :', score[0])
# print('CNN Test accuracy:', score[1])

# モデルを保存
model.save('identification_num.h5')