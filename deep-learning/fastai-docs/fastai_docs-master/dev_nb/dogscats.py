from nb_005 import *
torch.cuda.set_device(3)

PATH = Path('data/dogscats')
train_ds = ImageDataset.from_folder(PATH/'train')
valid_ds = ImageDataset.from_folder(PATH/'valid')
data_norm,data_denorm = normalize_funcs(*imagenet_stats)

arch,size,lr = tvm.resnet34, 224,3e-3

tfms = get_transforms(do_flip=True, max_rotate=10, max_zoom=1.2, max_lighting=0.3, max_warp=0.15)
tds = transform_datasets(train_ds, valid_ds, tfms, size=size)
data = DataBunch.create(*tds, bs=64, num_workers=8, tfms=data_norm)

learn = ConvLearner(data, arch, 2, metrics=accuracy)
learn.fit_one_cycle(2, slice(lr))

