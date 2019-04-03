import numpy as np
import models
from keras.callbacks import History
from scipy import io
history = History()


np.random.seed(111)  # for reproducibility

from scipy import io


SVHN_train = io.loadmat(r'C:\Users\Baruch\PycharmProjects\Try_Keras\train_32x32.mat')
SVHN_test = io.loadmat(r'C:\Users\Baruch\PycharmProjects\Try_Keras\test_32x32.mat')

X_train = SVHN_train['X'].transpose(3, 0, 1, 2)
X_test = SVHN_test['X'].transpose(3, 0, 1, 2)
y_train = (SVHN_train['y']-1).reshape(SVHN_train['y'].shape[0])
y_test = (SVHN_test['y']-1).reshape(SVHN_test['y'].shape[0])

# Preprocess input data

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 255
X_test /= 255

# rgb2gray
X_train = np.dot(X_train[..., :3], [0.299, 0.587, 0.114])
X_test = np.dot(X_test[..., :3], [0.299, 0.587, 0.114])

X_train_conv = X_train.reshape((X_train.shape[0], 32, 32, 1))
X_test_conv = X_test.reshape((X_test.shape[0], 32, 32, 1))
X_train = X_train.reshape((X_train.shape[0], 1024))
X_test = X_test.reshape((X_test.shape[0], 1024))

svhn_model = models.Conv2D_SVHN_model()

epochs = 100
batch_size = 64

hist = svhn_model.fit(X_train_conv, X_train, epochs=epochs, batch_size=batch_size)
