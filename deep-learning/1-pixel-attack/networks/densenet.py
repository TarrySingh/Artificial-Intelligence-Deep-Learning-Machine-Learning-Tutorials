import keras
import math
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras import optimizers
from keras import regularizers
from keras.utils import plot_model

from networks.train_plot import PlotLearning

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class DenseNet:
    def __init__(self, epochs=250, batch_size=64, load_weights=True):
        self.name               = 'densenet'
        self.model_filename     = 'networks/models/densenet.h5'
        self.growth_rate        = 12 
        self.depth              = 100
        self.compression        = 0.5
        self.num_classes        = 10
        self.img_rows, self.img_cols = 32, 32
        self.img_channels       = 3
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = 782
        self.weight_decay       = 0.0001
        self.log_filepath       = r'networks/models/densenet/'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)
    
    def count_params(self):
        return self._model.count_params()

    def color_preprocessing(self, x_train,x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        return x_train, x_test

    def scheduler(self, epoch):
        if epoch <= 75:
            return 0.1
        if epoch <= 150:
            return 0.01
        if epoch <= 210:
            return 0.001
        return 0.0005

    def densenet(self, img_input,classes_num):

        def bn_relu(x):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        def bottleneck(x):
            channels = self.growth_rate * 4
            x = bn_relu(x)
            x = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            x = bn_relu(x)
            x = Conv2D(self.growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            return x

        def single(x):
            x = bn_relu(x)
            x = Conv2D(self.growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            return x

        def transition(x, inchannels):
            outchannels = int(inchannels * self.compression)
            x = bn_relu(x)
            x = Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            x = AveragePooling2D((2,2), strides=(2, 2))(x)
            return x, outchannels

        def dense_block(x,blocks,nchannels):
            concat = x
            for i in range(blocks):
                x = bottleneck(concat)
                concat = concatenate([x,concat], axis=-1)
                nchannels += self.growth_rate
            return concat, nchannels

        def dense_layer(x):
            return Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay))(x)


        nblocks = (self.depth - 4) // 6 
        nchannels = self.growth_rate * 2

        x = Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(img_input)

        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)
        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)
        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)
        x = bn_relu(x)
        x = GlobalAveragePooling2D()(x)
        x = dense_layer(x)
        return x

    def train(self):
        # load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test  = keras.utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test  = x_test.astype('float32')
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        # build network
        img_input = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
        output    = self.densenet(img_input,self.num_classes)
        model     = Model(img_input, output)
        model.summary()
        
        # plot_model(model, show_shapes=True, to_file='model.png')

        # set optimizer
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # set callback
        tb_cb     = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(self.scheduler)
        ckpt      = ModelCheckpoint(self.model_filename, monitor='val_loss', verbose=0, save_best_only= True, mode='auto')
        plot_callback = PlotLearning()
        cbks      = [change_lr,tb_cb,ckpt, plot_callback]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

        datagen.fit(x_train)

        # start training
        model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size), steps_per_epoch=self.iterations, epochs=self.epochs, callbacks=cbks,validation_data=(x_test, y_test))
        model.save(self.model_filename)

        self.param_count = self._model.count_params()
        self._model = model

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._model.evaluate(x_test, y_test, verbose=0)[1]