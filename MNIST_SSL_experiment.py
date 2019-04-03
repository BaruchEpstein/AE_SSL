from keras.datasets import mnist
import numpy as np
import models
from keras.models import Model, load_model

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess input data
#X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
#X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# scale to [0,1]
#X_train /= 255
X_test /= 255

# prepare data as input to convolutional model
X_test_conv = X_test.reshape((X_test.shape[0], 28, 28, 1))
#X_train_conv = X_train.reshape((X_train.shape[0], 28, 28, 1))

conv_model = load_model(r'Experiments/Snapshots/unsupervised_MNIST_conv_model.h5')

# model test outputs
test_pred_conv = conv_model.predict(X_test_conv)

# model abs errors
test_abs_diff_conv = np.abs(X_test - test_pred_conv)

# model mean L2 error
mu_conv = np.linalg.norm(test_abs_diff_conv) / 100
print('mu_conv = ' + str(mu_conv))

# set eps and find G_eps
eps = 0.25
G_eps = np.sqrt(np.sum((test_abs_diff_conv**2), 1)) - mu_conv < mu_conv*eps
size_G_eps = np.sum(G_eps)
#X_G_eps = X_test[G_eps]
#X_conv_G_eps = X_test_conv[G_eps]

conv_model_encoder = Model(conv_model.layers[0].get_input_at(0), conv_model.layers[8].get_output_at(0))
X_test_codes = conv_model_encoder.predict(X_test_conv)

# find eta, eta_eps, eta', C, C_eps
eta = np.inf
eta_eps = np.inf
eta_prime = np.inf
C, C_eps = 0, 0

for ind1 in range(0, X_test.shape[0]):
    for ind2 in range(ind1+1, X_test.shape[0]):
        if y_test[ind1] != y_test[ind2]:
            d = np.linalg.norm(X_test[ind1] - X_test[ind2])
            d_code = np.linalg.norm(X_test_codes[ind1] - X_test_codes[ind2])
            d_rec = np.linalg.norm(test_pred_conv[ind1] - test_pred_conv[ind2])
            if d < eta:
                eta = d
            c = d_rec / d_code
            if c > C:
                C = c
            if G_eps[ind1] and G_eps[ind2]:
                if d < eta_eps:
                    eta_eps = d
                if d_code < eta_prime:
                    eta_prime = d_code
                if c > C_eps:
                    C_eps = c

print([eta, eta_eps, eta_prime, C, C_eps])


