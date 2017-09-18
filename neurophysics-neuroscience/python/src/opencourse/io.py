import os.path as op
import os
import mne

datasets = {'stevenson_v2': 'https://www.dropbox.com/s/a1ado3q51maxvct/StevensonV2.mat',
            'stevenson_v4': 'https://www.dropbox.com/s/s3y05ozij1qwof9/StevensonV4.mat',
            'connectivity': 'https://www.dropbox.com/s/ce9di12cibi85og/matrices_connectivity.mat'}


def download_file(key):
    if key not in datasets.keys():
        raise ValueError('key must be one of {}'.format(datasets.keys()))
    if not op.exists('../../data'):
        os.makedirs('../../data')
    url = datasets[key]
    name = op.basename(url)
    file_path = '../../data/{}'.format(name)
    _ = mne.utils._fetch_file(url, file_path)
    abs_path = op.abspath(file_path)
    print('Saved to: {}'.format(abs_path))


def download_all_files():
    for key in datasets.keys():
        download_file(key)
