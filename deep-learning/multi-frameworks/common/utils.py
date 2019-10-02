import os
import pathlib
import subprocess
import sys
import glob
import tarfile
import pickle
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection import train_test_split

if sys.version_info.major == 2:
    # Backward compatibility with python 2.
    from six.moves import urllib
    urlretrieve = urllib.request.urlretrieve
else:
    from urllib.request import urlretrieve


def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def get_cuda_version():
    """Get CUDA version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\version.txt"
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        path = '/usr/local/cuda/version.txt'
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = f.read().replace('\n','')
        return data
    else:
        return "No CUDA in this machine"

def get_cudnn_version():
    """Get CUDNN version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #candidates = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\include\\cudnn.h"]
    elif sys.platform == 'linux':
        candidates = ['/usr/include/x86_64-linux-gnu/cudnn_v[0-99].h',
                      '/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    elif sys.platform == 'darwin':
        candidates = ['/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    for c in candidates:
        file = glob.glob(c)
        if file: break
    if file:
        with open(file[0], 'r') as f:
            version = ''
            for line in f:
                if "#define CUDNN_MAJOR" in line:
                    version = line.split()[-1]
                if "#define CUDNN_MINOR" in line:
                    version += '.' + line.split()[-1]
                if "#define CUDNN_PATCHLEVEL" in line:
                    version += '.' + line.split()[-1]
        if version:
            return version
        else:
            return "Cannot find CUDNN version"
    else:
        return "No CUDNN in this machine"



def read_batch(src):
    '''Unpack the pickle files
    '''
    with open(src, 'rb') as f:
        if sys.version_info.major == 2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
    return data


def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y


def yield_mb(X, y, batchsize=64, shuffle=False):
    if shuffle:
        X, y = shuffle_data(X, y)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]

        
def yield_mb_X(X, batchsize):
    """ Function yield (complete) mini_batches of data"""
    for i in range(len(X)//batchsize):
        yield i, X[i*batchsize:(i+1)*batchsize]

        
def yield_mb_tn(X, y, batchsize=64, shuffle=False):
    """ Function yields mini-batches for time-series, layout=TN """
    if shuffle:
        X, y = shuffle_data(X, y)
    # Reshape
    X = np.swapaxes(X, 0, 1)
    # Only complete batches are submitted
    for i in range(X.shape[-1] // batchsize):
        yield X[..., i*batchsize:(i + 1)*batchsize], y[i * batchsize:(i + 1) * batchsize]
    
    
def give_fake_data(batches):
    """ Create an array of fake data to run inference on"""
    np.random.seed(0)
    dta = np.random.rand(batches, 224, 224, 3).astype(np.float32)
    return dta, np.swapaxes(dta, 1, 3)


def process_cifar():
    '''Load data into RAM'''
    print('Preparing train set...')
    train_list = [read_batch('./cifar-10-batches-py/data_batch_{0}'.format(i + 1)) for i in range(5)]
    x_train = np.concatenate([t['data'] for t in train_list])
    y_train = np.concatenate([t['labels'] for t in train_list])
    print('Preparing test set...')
    tst = read_batch('./cifar-10-batches-py/test_batch')
    x_test = tst['data']
    y_test = np.asarray(tst['labels'])
    return x_train, x_test, y_train, y_test


def maybe_download_cifar(src="https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/cifar-10-python.tar.gz"):
    '''Load the training and testing data
    Mirror of: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    '''
    try:
        return process_cifar()
    except:
        # Catch the exception that file doesn't exist
        # Download
        print('Data does not exist. Downloading ' + src)
        fname, h = urlretrieve(src, './delete.me')
        print('Extracting files...')
        with tarfile.open(fname) as tar:
            tar.extractall()
        os.remove(fname)
        return process_cifar()


def process_imdb():
    '''Load data into RAM'''
    with np.load('imdb.npz') as f:
        print('Preparing train set...')
        x_train, y_train = f['x_train'], f['y_train']
        print('Preparing test set...')
        x_test, y_test = f['x_test'], f['y_test']
    return x_train, x_test, y_train, y_test


def maybe_download_imdb(src="https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/imdb.npz"):
    '''Load the training and testing data
    Mirror of: https://s3.amazonaws.com/text-datasets/imdb.npz'''
    try:
        return process_imdb()
    except:
        # Catch exception that file doesn't exist
        # Download
        print('Data does not exist. Downloading ' + src)
        fname, h = urlretrieve(src, './imdb.npz')
        # No need to extract
        x_train, x_test, y_train, y_test = process_imdb()
        return x_train, x_test, y_train, y_test


def cifar_for_library(channel_first=True, one_hot=False):
    # Raw data
    x_train, x_test, y_train, y_test = maybe_download_cifar()
    # Scale pixel intensity
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Reshape
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)
    # Channel last
    if not channel_first:
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
    # One-hot encode y
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test


def imdb_for_library(seq_len=100, max_features=20000, one_hot=False):
    ''' Replicates same pre-processing as:
    https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py

    I'm not sure if we want to load another version of IMDB that has got 
    words, but if it does have words we would still convert to index in this 
    backend script that is not meant for others to see ...    

    But I'm worried this obfuscates the data a bit?
    '''
    # 0 (padding), 1 (start), 2 (OOV)
    START_CHAR = 1
    OOV_CHAR = 2
    INDEX_FROM = 3
    # Raw data (has been encoded into words already)
    x_train, x_test, y_train, y_test = maybe_download_imdb()
    # Combine for processing
    idx = len(x_train)
    _xs = np.concatenate([x_train, x_test])
    # Words will start from INDEX_FROM (shift by 3)
    _xs = [[START_CHAR] + [w + INDEX_FROM for w in x] for x in _xs]
    # Max-features - replace words bigger than index with oov_char
    # E.g. if max_features = 5 then keep 0, 1, 2, 3, 4 i.e. words 3 and 4
    if max_features:
        print("Trimming to {} max-features".format(max_features))
        _xs = [[w if (w < max_features) else OOV_CHAR for w in x] for x in _xs]
        # Pad to same sequences
    print("Padding to length {}".format(seq_len))
    xs = np.zeros((len(_xs), seq_len), dtype=np.int)
    for o_idx, obs in enumerate(_xs):
        # Match keras pre-processing of taking last elements
        obs = obs[-seq_len:]
        for i_idx in range(len(obs)):
            if i_idx < seq_len:
                xs[o_idx][i_idx] = obs[i_idx]
    # One-hot
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = np.array(xs[:idx]).astype(np.int32)
    x_test = np.array(xs[idx:]).astype(np.int32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test


def resize_im(im_loc, size=(264, 264)):
    # Size 264 allows centre-crop of 224 for image-augmentation
    im = Image.open(im_loc).resize(size, Image.BILINEAR).convert('RGB')
    im.save(im_loc)


def resize_chestxray_mp(im_list):
    print("{} images will be resized to (264,264)".format(len(im_list)))
    print("Images will be overwritten to save disk-space")
    pool = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())
    pool.map(resize_im, im_list)

    
def download_data_chextxray(csv_dest, base_url = 'https://ikpublictutorial.blob.core.windows.net/'):
                            
    # Check whether files-exist
    try:
        df = pd.read_csv(os.path.join(csv_dest, "Data_Entry_2017.csv"))
        img_dir = os.path.join(csv_dest, "images")
        img_locs = df['Image Index'].map(lambda im: os.path.join(img_dir, im)).values
        for im in img_locs:
            assert os.path.isfile(im)
        print("Data already exists")
    except Exception as err:
        print("Data does not exist")
        
        # Locations
        CSV_URL = base_url + 'deeplearningframeworks/Data_Entry_2017.csv'
        CONTAINER_URL  = base_url + 'chest'        
        container_dest = os.path.join(csv_dest , 'images')
        
        # Create full directory recursively
        print("Creating data directory")
        pathlib.Path(container_dest).mkdir(parents=True, exist_ok=True) 
        
        # Download labels 
        print("Downloading CSV file with labels ...")
        subprocess.call(['wget', '-N', CSV_URL, '-P', csv_dest])
        
        # Download Images
        print("Downloading Chext X-ray images ...")
        print("This requires AzCopy:")
        print("https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-linux")
        print("This requires 45 GB of free-space, and may take at-least 10 minutes")
        print("You may have to increase the size of your OS Disk in Azure Portal")
        subprocess.call(['azcopy', '--source', CONTAINER_URL, 
                         '--destination', container_dest, '--quiet', '--recursive'])
        print("Data Download Complete")
        print("About to overwrite images: Image.open(loc).resize((264, 264), Image.BILINEAR).convert('RGB')")
        # Get image locations for resize
        df = pd.read_csv(os.path.join(csv_dest, "Data_Entry_2017.csv"))
        img_dir = os.path.join(csv_dest, "images")
        img_locs = df['Image Index'].map(lambda im: os.path.join(img_dir, im)).values
        resize_chestxray_mp(img_locs)
        print("Finished resizing")  
        
def get_imgloc_labels(img_dir, lbl_file, patient_ids):
    """ Function to process data into a list of img_locs containing string paths
    and labels, which are one-hot encoded."""
    # Read labels-csv
    df = pd.read_csv(lbl_file)
    # Process
    # Split labels on unfiltered data
    df_label = df['Finding Labels'].str.split(
        '|', expand=False).str.join(sep='*').str.get_dummies(sep='*')
    # Filter by patient-ids (both)
    df_label['Patient ID'] = df['Patient ID']
    df_label = df_label[df_label['Patient ID'].isin(patient_ids)]
    df = df[df['Patient ID'].isin(patient_ids)]
    # Remove unncessary columns
    df_label.drop(['Patient ID','No Finding'], axis=1, inplace=True)  

    # List of images (full-path)
    img_locs =  df['Image Index'].map(lambda im: os.path.join(img_dir, im)).values
    # One-hot encoded labels (float32 for BCE loss)
    labels = df_label.values   
    return img_locs, labels


def compute_roc_auc(data_gt, data_pd, classes, full=True):
    roc_auc = []
    for i in range(classes):
        roc_auc.append(roc_auc_score(data_gt[:, i], data_pd[:, i]))
    print("Full AUC", roc_auc)
    roc_auc = np.mean(roc_auc)
    return roc_auc

def get_mxnet_model(prefix, epoch):
    """Download an MXNet model if it doesn't exist"""
    def download(url):
        filename = url.split("/")[-1]
        if not os.path.exists(filename):
            urlretrieve(url, filename)
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

def get_train_valid_test_split(n, train=0.7, valid=0.1, test=0.2, shuffle=False):
    other_split = valid+test
    if train+other_split!=1:
        raise ValueError("Train, Valid, Test splits should sum to 1")
    train_set, other_set = train_test_split(range(1,n+1), 
                                            train_size=train, test_size=other_split, shuffle=shuffle)
    valid_set, test_set = train_test_split(other_set, 
                                           train_size=valid/other_split, 
                                           test_size=test/other_split,
                                           shuffle=False)
    print("train:{} valid:{} test:{}".format(len(train_set), len(valid_set), len(test_set)))
    return train_set, valid_set, test_set
