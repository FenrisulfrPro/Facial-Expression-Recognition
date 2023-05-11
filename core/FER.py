import tensorflow.keras
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation,Dropout,Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

#batch_size = 64 #修改
#epochs = 20
batch_siz = 128
epochs = 200
num_classes = 7  # 表情的类别数目
x_train, y_train, x_test, y_test = [], [], [], []


def concert_fer2013():
    global x_train, y_train, x_test, y_test
    path = r"C:\Users\fen\Desktop\Graduation project\FER-master\data\fer2013\fer2013.csv"
    data = pd.read_csv(path)
    num_of_instances = len(data)  # 获取数据集的数量
    print("数据集的数量为：", num_of_instances)
    pixels = data['pixels']
    emotions = data['emotion']
    usages = data['Usage']
    for emotion, img, usage in zip(emotions, pixels, usages):
        try:
            emotion = tensorflow.keras.utils.to_categorical(emotion, num_classes)  # 独热向量编码
            val = img.split(" ")
            pixels = np.array(val, 'float32')

            if (usage == 'Training'):
                x_train.append(pixels)
                y_train.append(emotion)
            elif (usage == 'PublicTest'):
                x_test.append(pixels)
                y_test.append(emotion)
        except:
            print("", end="")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 48, 48, 1)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = x_test.reshape(-1, 48, 48, 1)


class Model:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (1, 1), strides=1, padding='same', input_shape=(48, 48, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(2048))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    def train_model(self):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        val_datagen = ImageDataGenerator(
            rescale=1. / 255)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        history = LossHistory()
        history_fit = self.model.fit_generator(train_datagen.flow(x_train, y_train, batch_siz),
                                          steps_per_epoch=len(x_train)//batch_siz,
                                          epochs=epochs,
                                          validation_data=val_datagen.flow(x_test, y_test, batch_siz),
                                          validation_steps=2000,
                                          callbacks=[history]
                                          )

        # loss = history_fit.history['loss']
        # val_loss = history_fit.history['val_loss']
        # epochs = range(1, len(loss) + 1)
        # plt.plot(epochs, loss, label='Training loss')
        # plt.plot(epochs, val_loss, label='Validation loss')
        # plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

        # 进行训练
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        train_score = self.model.evaluate(train_datagen.flow(x_train, y_train, batch_siz), verbose=0)
        print('Train loss:', train_score[0])
        print('Train accuracy:', 100 * train_score[1])

        test_score = self.model.evaluate(val_datagen.flow(x_test, y_test, batch_siz), verbose=0)
        print('Test loss:', test_score[0])
        print('Test accuracy:', 100 * test_score[1])
        history.loss_plot('epoch')

    def save_model(self):
        self.model.save('my_model.h5')


class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_accuracy = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_accuracy['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_accuracy['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_accuracy[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


if __name__=='__main__':
    concert_fer2013()
    model=Model()
    model.build_model()
    print('model built')
    model.train_model()
    print('model trained')
    model.save_model()
    print('model saved')

