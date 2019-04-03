from keras.datasets import mnist
import numpy as np
import models

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# scale to [0,1] and binarize
X_train /= 255
X_test /= 255

X_train = np.round(X_train)
X_test = np.round(X_test)

# define model
model = models.FC_MNIST_model(input_shape=(784,), layer_sizes=[200, 30, 200])

# training
epochs = 10
batch_size = 256

steps = 10

gammas = np.arange(0.05, 0.5, 0.05)
gamma_train_errors = np.zeros((len(gammas), steps))
gamma_test_errors = np.zeros((len(gammas), steps))
Lipshitz_bounds = np.zeros((1, steps))
mu_values = np.zeros((1, steps))

for step in range(0, steps):
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
    filepath = r'Experiments/Snapshots/unsupervised_MNIST' + str(step*10) + '_model.h5'
    model.save(filepath)
    train_pred = model.predict(X_train)
    train_abs_diff = np.abs(X_train - train_pred)
    test_pred = model.predict(X_test)
    test_abs_diff = np.abs(X_test - test_pred)
    for gamma_ind, gamma in enumerate(gammas):
        gamma_train_errors[gamma_ind, step] = np.sum(train_abs_diff > (0.5 - gamma)) / (784.0*60000)
        gamma_test_errors[gamma_ind, step] = np.sum(test_abs_diff > (0.5 - gamma)) / (784.0 * 10000)

    Lipshitz_bound = 1
    for layer in model.layers:
        Lipshitz_bound *= np.linalg.norm(layer.get_weights()[0], 2)
    Lipshitz_bounds[0, step] = Lipshitz_bound

    mu_values[0, step] = np.linalg.norm(test_abs_diff) / 100


