# 模型训练
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import os,shutil
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

train_dir=r'C:\Users\fen\Desktop\Graduation project\FER-master\data\datasets\TrainDataAdd' #增强后的训练集
val_dir=r'C:\Users\fen\Desktop\Graduation project\FER-master\data\datasets\val'
test_dir=r'C:\Users\fen\Desktop\Graduation project\FER-master\data\datasets\test'

train_datagen=ImageDataGenerator()
test_datagen=ImageDataGenerator()

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=128,
    class_mode='categorical'
)
validation_generator=test_datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    batch_size=128,
    class_mode='categorical'
)

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=128,
    class_mode='categorical'
)

#搭建神经网络
#kernel_regularizer=regularizers.l2(0.01),
model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 padding='same',input_shape=(48,48,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))

#训练

#train_generator = train_generator.repeat()
#validation_generator = validation_generator.repeat()

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.RMSprop(lr=0.0001),  #need to train a lot of epochs
              metrics=['accuracy'])

history=model.fit_generator(
    train_generator,
    #steps_per_epoch=452,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=32
)



#保存
model.save('my_model1.h5')

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)
plt.figure("acc")
plt.plot(epochs,acc,'r-',label='Training acc')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('The comparision of train_acc and val_acc')
plt.legend()
plt.show()

plt.figure("loss")
plt.plot(epochs,loss,'r-',label='Training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.title('The comparision of train_loss and val_loss')
plt.legend()
plt.show()
