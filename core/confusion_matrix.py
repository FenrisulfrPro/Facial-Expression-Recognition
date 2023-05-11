from sklearn.metrics import confusion_matrix
import itertools
import tensorflow.keras
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))


if __name__=='__main__':
    concert_fer2013()
    model = load_model('my_model.h5')
    plot_confuse(model, x_test, y_test)
