import os, math, csv
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from pandas import read_csv

def load_cifar_10():
    from keras.datasets import cifar10
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train,y_train),(x_test,y_test)

def load_cifar_100():
    from keras.datasets import cifar100
    num_classes = 100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train,y_train),(x_test,y_test)

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def initializer():
    if not os.path.exists('results/'):
        os.mkdir('results')
    if not os.path.exists('weights/'):
        os.mkdir('weights')

def plot_log(filename, show=True):
    # load data
    log_df = read_csv(filename)
    # epoch_list = [i for i in range(len(values[:,0]))]
    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for column in list(log_df):
        if 'loss' in column and 'val' in column:
            plt.plot(log_df['epoch'].tolist(),log_df[column].tolist(), label=column)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for column in list(log_df):
        if 'acc' in column :
            plt.plot(log_df['epoch'].tolist(),log_df[column].tolist(), label=column)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def data_generator(x,y,batch_size):
    x_train,y_train = x,y
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    generator = datagen.flow(x_train,y_train,batch_size=batch_size)
    while True:
        x,y  = generator.next()
        yield ([x,y],[y,x])