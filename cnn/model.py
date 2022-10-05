import IPython
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers
from keras.utils import to_categorical
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, TimeDistributed, LSTM, ReLU, Bidirectional, GRU, \
    MaxPool1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import tensorflow as tf
# import kerastuner as kt
import keras_tuner as kt
from tensorflow import keras
from keras.layers import Dense, Flatten



input_data = pd.read_csv('./data/train_data_even_odd.csv')
train_x = input_data.drop('label', axis=1)
train_y = input_data['label']

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

train_shape = train_x.shape
print(train_x.shape)
print(train_y.shape)

train_x = np.array_split(train_x,4,axis=1)
train_ax, train_ay, train_az, train_str = train_x[0], train_x[1], train_x[2], train_x[3]


train_x = np.zeros((train_shape[0], 32, 4))
train_x[..., 0] = train_ax
train_x[..., 1] = train_ay
train_x[..., 2] = train_az
train_x[..., 3] = train_str

del train_ax, train_ay, train_az, train_str



input_data = pd.read_csv('./data/test_data_even_odd.csv')
test_x = input_data.drop('label', axis=1)
test_y = input_data['label']

test_x = test_x.to_numpy()
test_y = test_y.to_numpy()

test_shape = test_x.shape
print(test_x.shape)
print(test_y.shape)

test_x = np.array_split(test_x,4,axis=1)
test_ax, test_ay, test_az, test_str = test_x[0], test_x[1], test_x[2], test_x[3]


test_x = np.zeros((test_shape[0], 32, 4))
test_x[..., 0] = test_ax
test_x[..., 1] = test_ay
test_x[..., 2] = test_az
test_x[..., 3] = test_str
del test_ax, test_ay, test_az, test_str

train_y = to_categorical(train_y, num_classes=8)
test_y = to_categorical(test_y, num_classes=8)

model = Sequential()

#model_1
model.add(Conv1D(filters=32 , kernel_size=3, activation='relu', input_shape=(32, 4)))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding = 'same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=8, kernel_size=3, activation='relu', padding = 'same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(8, activation='softmax'))


# Load Weights
model = Model(inputs = model.input, outputs = model.output)


model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'])


model.summary()


history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                        epochs=10000, batch_size=128,  verbose=1)


model.save('odd_even_model.h5')


(loss, accuracy) = model.evaluate(test_x,test_y, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


y_pred_train = model.predict(train_x)

max_y_pred_train = np.argmax(y_pred_train, axis=1)


plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.legend(['train', 'validation'])

# confusion matrix
LABELS = ['0',
          '1',
          '2',
          '3',
          '4',
          '5',
          '6',
          '7']
y_pred_test = model.predict(test_x)

max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(test_y, axis=1)

matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(matrix,
            cmap='PuOr',
            linecolor='white',
            linewidths=1,
            xticklabels=LABELS,
            yticklabels=LABELS,
            annot=True,
            fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
