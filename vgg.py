
from keras.models import *
from keras.layers import *
from keras.optimizers import *



def vggnet(input_size=(64,64,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    conv2 = Conv2D(128, 3, activation='relu',strides=2, padding='same', kernel_initializer='he_normal')(conv1)
    conv2=Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)


    conv3 = Conv2D(256, 3, activation='relu',strides=2, padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    conv4 = Conv2D(256, 3, activation='relu', strides=2,padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    flatten=Flatten()(conv4)
    dense4= Dense(1024,activation='relu')(flatten)
    dense5=Dense(2,activation='softmax')(dense4)
    model = Model(input=inputs, output=dense5)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
