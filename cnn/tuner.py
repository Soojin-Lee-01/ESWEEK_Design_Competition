import IPython
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense
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

# def split_data(dataset):
#     """ 데이터셋 나누기
#
#     Parameters
#     ----------
#     dataset : 데이터셋
#
#     Returns
#     -------
#     (데이터, 레이블)
#     """
#
#     total_value = []
#
#     dataset = dataset.transpose()
#     value = dataset[:-1]
#     label = dataset[-1:]
#     label = np.array(label.transpose())
#     # label = label.reshape(-1)
#
#     for i in range(len(value.columns)):
#         v_list = list(np.array(value[i].tolist()))
#         total_value.append(v_list)
#
#     total_value = np.array(total_value)
#
#     return total_value, label
#
#
# # 데이터셋파일 불러오기
# train_xy = pd.read_csv('./data/train_data.csv',  encoding='euc-kr')
# test_xy = pd.read_csv('./data/test_data.csv',  encoding='euc-kr')
#
# # 데이터와 라벨 분류
# train_x, train_y = split_data(train_xy)
# test_x, test_y = split_data(test_xy)
#
#
#
# train_x = np.array(train_x).reshape((-1, 256, 1))
# test_x = np.array(test_x).reshape((-1, 256, 1))
#
#
# train_y = to_categorical(train_y, num_classes=8)
# test_y = to_categorical(test_y, num_classes=8)


# train 파일 불러오기
input_data = pd.read_csv('./data/train_data.csv' , encoding='utf8')
input_data.head()

train_x = input_data.drop('label', axis=1)
train_y = input_data['label']

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

train_shape = train_x.shape
print(train_x.shape)
print(train_y.shape)

train_x = np.array_split(train_x,4,axis=1)
train_ax, train_ay, train_az, train_str = train_x[0], train_x[1], train_x[2], train_x[3]


train_x = np.zeros((train_shape[0], 64, 4))
train_x[..., 0] = train_ax
train_x[..., 1] = train_ay
train_x[..., 2] = train_az
train_x[..., 3] = train_str

del train_ax, train_ay, train_az, train_str


# test 파일 불러오기
input_data = pd.read_csv('./data/test_data.csv', encoding='utf8' )
input_data.head()

test_x = input_data.drop('label', axis=1)
test_y = input_data['label']

test_x = test_x.to_numpy()
test_y = test_y.to_numpy()

test_shape = test_x.shape
print(test_x.shape)
print(test_y.shape)

test_x = np.array_split(test_x,4,axis=1) # 열 64개씩 끊어 4개로 분리
test_ax, test_ay, test_az, test_str = test_x[0], test_x[1], test_x[2], test_x[3]

# 4개로 분리해놓은 넘파이 배열을 4개의 채널(64*6)을 가진 넘파이 배열로 합침
test_x = np.zeros((test_shape[0], 64, 4))
test_x[..., 0] = test_ax
test_x[..., 1] = test_ay
test_x[..., 2] = test_az
test_x[..., 3] = test_str
del test_ax, test_ay, test_az, test_str

train_y = to_categorical(train_y, num_classes=8)
test_y = to_categorical(test_y, num_classes=8)

print(train_x.shape)
print(test_x.shape)
print(test_y.shape)
print(train_y.shape)


def model_builder(hp):
    model = Sequential()
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_units = hp.Int('units', min_value = 4, max_value = 50, step = 32)
    hp_units1 = hp.Int('units1', min_value=4, max_value=50, step=32)
    hp_drop = hp.Float('dropout_rate',min_value=0.0, max_value=0.5, step=0.05)
    hp_con2 = hp.Int('l2_num_filters', min_value=4, max_value=64)
    hp_con1 = hp.Int('l1_num_filters', min_value=4, max_value=64)
    model.add(Conv1D(filters=hp_con1 , kernel_size=3, activation='relu', input_shape=(64, 4)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=hp_con2, kernel_size=3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(units=hp_units, activation='relu'))
    model.add(Dense(units=hp_units1, activation='relu'))
    model.add(Dense(8, activation='softmax'))



    model = Model(inputs = model.input, outputs = model.output)


    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate), metrics=['accuracy'])

    return model


tuner = kt.Hyperband(model_builder,
                        'val_accuracy',
                        max_epochs=1000,
                        directory='my_dir',
                        project_name='intro1')





class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

tuner.search(train_x, train_y, epochs=1000, validation_data=(test_x, test_y), callbacks=[ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters()[0]
print(f"""
    하이퍼 파라미터 검색이 완료되었습니다. 
    최적화된 첫 번째 Dense 노드 수는 {best_hps.get('units')} 입니다.
    최적화된 첫 번째 Dense 노드 수는 {best_hps.get('units1')} 입니다.
    {best_hps.get('l1_num_filters')}
    {best_hps.get('l2_num_filters')}
    최적의 학습 속도는 {best_hps.get('learning_rate')} 입니다.
""")


model = tuner.hypermodel.build(best_hps)

model.summary()



history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                        epochs=1000, batch_size=32,  verbose=1)


model.save('tuner_all.h5')

(loss, accuracy) = model.evaluate(test_x,test_y, batch_size=32, verbose=1)
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
# Take the class with the highest probability from the test predictions
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

