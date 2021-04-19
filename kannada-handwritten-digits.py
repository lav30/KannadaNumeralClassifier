import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df_train = pd.read_csv('train.csv')
df_val = pd.read_csv('Dig-MNIST.csv')
df_test = pd.read_csv('test.csv')

img_size = 28
n_channels = 1

df_train = df_train.append(df_val)

X_train = df_train.drop(['label'], axis = 1) # dropping the label column from the features matrix
y_train = df_train['label'] # label is to be predicted

X_pred = df_test.drop(['id'], axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size =0.1) # validation set = 10 %

# Normalizing images

X_train, X_test, X_pred = X_train.apply(lambda x: x/255), X_test.apply(lambda x: x/255), X_pred.apply(lambda x: x/255)
y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test)

## Reshaping Images

X_train = X_train.values.reshape(-1, img_size, img_size, n_channels)
X_test = X_test.values.reshape(-1, img_size, img_size, n_channels)

y_train = y_train.to_numpy()
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

datagenerator = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=8,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.15,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None )

datagenerator.fit(X_train)

model = Sequential()  # MLP with CNN

# Feature Learning

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  # pooling layer 1
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu'))
model.add(Dropout(rate=0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  # pooling layer 2
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu'))
model.add(Dropout(rate=0.3))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  # pooling layer 3
model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu'))
model.add(Dropout(rate=0.5))

# Classification

model.add(Flatten())

model.add(Dense(128, activation="relu"))  # Fully connected layer
model.add(Dropout(0.5))

model.add(Dense(64, activation = "relu")) # Fully connected layer
model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax")) # Output layer / Classifier

# Model Compilation

model.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

batch_size=128
epochs = 20

history = model.fit_generator(datagenerator.flow(X_train, y_train, batch_size = batch_size),
                              epochs = epochs, validation_data = (X_test,y_test),
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])

# Interface

def recognize_digit(image):
    image = image.reshape(-1,28,28, 1)  # add a batch dimension and flatten array
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

gr.Interface(fn=recognize_digit,
             inputs="sketchpad",
             outputs=gr.outputs.Label(num_top_classes=3),
             ).launch();

