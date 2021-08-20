from data import datagen
from keras import backend as K
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from vgg import vggnet
from keras.callbacks import ModelCheckpoint



filepath="best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    X_train,Y_train,X_tests,Y_tests,X_testss,Y_testss=datagen()
    Y_train = to_categorical(Y_train, num_classes=2)
    Y_testss=to_categorical(Y_testss, num_classes=2)
    model = vggnet()
    model.fit(X_train, Y_train,validation_data=(X_testss, Y_testss),
              callbacks=callbacks_list,epochs=50, batch_size=32,shuffle=True)  # data是输入数据的X labels是Y
    for i in range(10):
        X_test=X_tests[i]
        Y_test=Y_tests[i]
        if (Y_test[0] == 0):
            print("A")
        else:
            print("B")
        Y_test = to_categorical(Y_test, num_classes=2)
        score=model.evaluate(X_test,Y_test)
        print(score)




