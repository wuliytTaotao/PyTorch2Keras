import torch
import torchvision
import cv2
import numpy as np

from torch.autograd import Variable

import tensorflow as tf
from tensorflow.keras import layers,regularizers


filename_test = 'data/dog2.png'

img = cv2.imread(filename_test)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# data pre-processing
img = cv2.resize(img, (227, 227))
img = img / 255.0
img = np.reshape(img, (1, 227, 227, 3))
# normalization，这是 PyTorch 预训练 AlexNet 模型的预处理方式，详情请见 https://pytorch.org/docs/stable/torchvision/models.html
mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3])
img = (img - mean) / std

# PyTorch
# PyTorch 数据输入 channel 排列和 Keras 不一致
img_tmp = np.transpose(img, (0, 3, 1, 2))

model = torchvision.models.alexnet(pretrained=True)

# torch.save(model, './model/alexnet.pth')
model = model.double()
model.eval()

y = model(Variable(torch.tensor(img_tmp)))
print(np.argmax(y.detach().numpy()))


# Keras
def get_AlexNet(num_classes=1000, drop_rate=0.5):
    """
    PyTorch 中实现的 AlexNet 预训练模型结构，filter 的深度分别为：（64，192，384，256，256）。
    返回 AlexNet 的 inputs 和 outputs
    """
    inputs = layers.Input(shape=[227, 227, 3])

    conv1 = layers.Conv2D(64, (11, 11), strides=(4, 4), padding='valid', activation='relu')(inputs)

    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

    conv2 = layers.Conv2D(192, (5, 5), strides=(1, 1), padding='same', activation='relu')(pool1)

    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)

    conv3 = layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)

    conv4 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    conv5 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)

    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv5)

    flat = layers.Flatten()(pool3)

    dense1 = layers.Dense(4096, activation='relu')(flat)
    dense1 = layers.Dropout(drop_rate)(dense1)
    dense2 = layers.Dense(4096, activation='relu')(dense1)
    dense2 = layers.Dropout(drop_rate)(dense2)
    outputs = layers.Dense(num_classes, activation='softmax')(dense2)

    return inputs, outputs


inputs, outputs = get_AlexNet()
model2 = tf.keras.Model(inputs, outputs)
model2.load_weights('./model/keras_alexnet.h5')
# prediction
print(np.argmax(model2.predict(img)))
