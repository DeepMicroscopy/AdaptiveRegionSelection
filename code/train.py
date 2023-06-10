from fastai.callbacks import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback
from tqdm import tqdm

from torch.utils.data import SubsetRandomSampler
from torchvision.models import mobilenet_v2
from utils import *
from SlideContainer import SlideContainer

np.seterr(divide='ignore', invalid='ignore')
warnings.simplefilter("ignore", UserWarning)
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()


def train(dataset_path, slide_name_list, level, patch_size, AL_annotations, stain_aug_p, stain_vector_pool, class_dict,
          batch_size, num_workers, pin_memory, device, max_lr, epochs, save_path):

    # step 1: load train and validation slides
    with open(AL_annotations, "r") as rf:
        annotation = json.load(rf)

    train_slides = []
    for name in tqdm(slide_name_list, desc="creating slideContainers (train)..."):
        if [anno for anno in annotation if (anno["file"] == name and anno["set_name"] == "train")]:
            train_slides.append(SlideContainer(dataset_path=dataset_path,
                                               slide_num=name.split(".")[0].split("_")[1],
                                               datatype="train_" + name.split("_")[0] + "_slide",
                                               class_dict=class_dict,
                                               mode="training",
                                               level=level,
                                               patch_size=patch_size,
                                               AL_annotations=AL_annotations,
                                               stain_aug_p=stain_aug_p,
                                               stain_vector_pool=stain_vector_pool))

    valid_slides = []
    for name in tqdm(slide_name_list, desc="creating slideContainers (validation)..."):
        if [anno for anno in annotation if (anno["file"] == name and anno["set_name"] == "valid")]:
            valid_slides.append(SlideContainer(dataset_path=dataset_path,
                                               slide_num=name.split(".")[0].split("_")[1],
                                               datatype="train_" + name.split("_")[0] + "_slide",
                                               class_dict=class_dict,
                                               mode="training",
                                               level=level,
                                               patch_size=patch_size,
                                               AL_annotations=AL_annotations,
                                               stain_aug_p=stain_aug_p,
                                               stain_vector_pool=stain_vector_pool))

    # step 2: create dataloader
    train_tfm = [dihedral_affine(), rotate(degrees=(-45, 45), p=0.75)]
    valid_tfm = None

    train_ds = ClassificationSlideDataset(train_slides, train_tfm, class_dict)
    valid_ds = ClassificationSlideDataset(valid_slides, valid_tfm, class_dict, is_valid=True)

    n_sample = 20  # each slide is sampled 20 times during each epoch (i.e., 20 patches that are randomly located)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True,
                          sampler=SubsetRandomSampler(sum([[i] * n_sample for i in range(len(train_slides))], [])),
                          num_workers=num_workers, pin_memory=pin_memory)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, drop_last=False,
                          sampler=SubsetRandomSampler(sum([[i] * n_sample for i in range(len(valid_slides))], [])),
                          num_workers=num_workers, pin_memory=pin_memory)

    dls = DataBunch(train_dl=train_dl, valid_dl=valid_dl, device=device)

    # debug: visualize input data
    for repeat in range(3):
        x, y = dls.one_batch(ds_type=DatasetType.Train)
        # print(f"x, y = dls.one_batch(ds_type=DatasetType.Train): x.shape={x.shape}, type(x)={type(x)}")
        # print(f"                                                 y.shape={y.shape}, type(y)={type(y)}")
        dls.train_ds.show_batch(x, y, save_path=os.path.join(save_path, 'plots', f"show_batch_train_{repeat}.png"))
        x, y = dls.one_batch(ds_type=DatasetType.Valid)
        # print(f"x, y = dls.one_batch(ds_type=DatasetType.Valid): x.shape={x.shape}, type(x)={type(x)}")
        # print(f"                                                 y.shape={y.shape}, type(y)={type(y)}")
        dls.valid_ds.show_batch(x, y, save_path=os.path.join(save_path, 'plots', f"show_batch_validation_{repeat}.png"))


    # step 3: build and train
    learn = cnn_learner(dls, mobilenet_v2, pretrained=True, metrics=accuracy)
    learn.loss_func = CrossEntropyFlat()
    learn.unfreeze()
    fit_one_cycle(learn=learn, cyc_len=epochs, max_lr=slice(max_lr),
                  callbacks=[
                      ShowGraph(learn),
                      SaveModelCallback(learn),
                      CountSamplingRateCallback(learn),
                      EarlyStoppingCallback(learn, patience=100)])

    learn.export(os.path.join(save_path, save_path + '.pkl'))
    learn.recorder.plot_lr(return_fig=True).savefig(os.path.join(save_path, 'plots', "plot_lr.png"))
    learn.recorder.plot_losses(return_fig=True).savefig(os.path.join(save_path, 'plots', "plot_losses.png"))
    learn.recorder.plot_metrics(return_fig=True).savefig(os.path.join(save_path, 'plots', "plot_metrics.png"))

    del learn
    del train_slides
    del valid_slides
    gc.collect()
    torch.cuda.empty_cache()
