from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Input, MaxPooling2D


def FC_MNIST_model(input_shape=(784), layer_sizes=[200, 30, 200]):

    model = Sequential()
    first_layer_flag = True
    for s in layer_sizes:
        if first_layer_flag:
            model.add(Dense(s, input_shape=input_shape, activation='relu'))
            first_layer_flag = False
        else:
            model.add(Dense(s, activation='relu'))
    model.add(Dense(input_shape[0], activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    return model


def Conv2D_MNIST_model():
    input = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, 5, activation='relu')(input)
    mp1   = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, 5, activation='relu')(mp1)
    mp2   = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(500, 4, activation='relu')(mp2)
    flat  = Flatten()(conv3)
    dense1 = Dense(250, activation='relu')(flat)
    dense2 = Dense(50, activation='relu')(dense1)
    dense3 = Dense(250, activation='relu')(dense2)
    dense4 = Dense(500, activation='relu')(dense3)
    dense5 = Dense(1000, activation='relu')(dense4)
    rec = Dense(784, activation='sigmoid')(dense5)

    model = Model(input, rec)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    return model

def large_Conv2D_MNIST_model():
    input = Input(shape=(28, 28, 1))
    conv1 = Conv2D(64, 5, activation='relu')(input)
    mp1   = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(128, 5, activation='relu')(mp1)
    mp2   = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(750, 4, activation='relu')(mp2)
    flat  = Flatten()(conv3)
    dense1 = Dense(250, activation='relu')(flat)
    dense2 = Dense(50, activation='relu')(dense1)
    dense3 = Dense(750, activation='relu')(dense2)
    dense4 = Dense(750, activation='relu')(dense3)
    dense5 = Dense(1000, activation='relu')(dense4)
    rec = Dense(784, activation='sigmoid')(dense5)

    model = Model(input, rec)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    return model



def Conv2D_SVHN_model():

    input = Input(shape=(32, 32, 1))
    conv1 = Conv2D(32, 5, activation='relu')(input)
    conv2 = Conv2D(64, 5, activation='relu')(conv1)
    mp2   = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(256, 5, activation='relu')(mp2)
    mp3   = MaxPooling2D((2, 2))(conv3)
    conv4 = Conv2D(512, 4, activation='relu')(mp3)
    flat  = Flatten()(conv4)
    dense1 = Dense(250, activation='relu')(flat)
    dense2 = Dense(50, activation='relu')(dense1)
    dense3 = Dense(250, activation='relu')(dense2)
    dense4 = Dense(500, activation='relu')(dense3)
    dense5 = Dense(1000, activation='relu')(dense4)
    rec = Dense(1024, activation='sigmoid')(dense5)

    model = Model(input, rec)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    return model
