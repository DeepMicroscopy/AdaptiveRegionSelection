import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from SlideContainer import *
from utils import *
from  result_analysis import get_pred_map

np.seterr(divide='ignore', invalid='ignore')
warnings.simplefilter("ignore", UserWarning)
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()


def predict(learner_path, slide_list, output_hdf5_filename, dataset_path, slide_level, patch_size,
            predict_level, class_dict, batch_size, num_workers, pin_memory, device, save_path=None):
    """ 1. inference for each slide in the (slide_list) at (slide_level) on the trained model (learner_filepath):
           first patch partition and prediction, then stitch into a map
        2. store logit results in (output_hdf5_filename)

        * predict_level l_p determines the resolution of the map:
            H_map = H_slide / (2 ** l_p)
            W_map = W_slide / (2 ** l_p)
    """
    print(f"predicting with {learner_path}...")
    print(f"predicting on {slide_list}...")

    # step 1: load model, inference slide
    state = torch.load(learner_path, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(
        learner_path)
    model = state.pop('model')
    model = model.to(device)

    for slide_i, slide_name in enumerate(slide_list):
        print(f"inference {slide_i + 1}/{len(slide_list)}: slide {slide_name}...")

        if os.path.exists(output_hdf5_filename):
            store = h5py.File(output_hdf5_filename, "r")
            key_list = list(store.keys())
            store.close()
            if slide_name + '_logits' in key_list:
                print(f"killed... (predicted already)")
                continue

        start = time.time()
        datatype = slide_name.split("_")[0]
        if datatype == "tumor" or datatype == "normal":
            datatype = "train_" + datatype + "_slide"
        elif datatype == "test":
            datatype = "test_slide"
        else:
            raise ValueError

        slide = SlideContainer(dataset_path=dataset_path,
                               slide_num=slide_name.split(".")[0].split("_")[1],
                               datatype=datatype,
                               class_dict=class_dict,
                               mode='inference',
                               level=slide_level,
                               patch_size=patch_size)

        # step 2: patch partition
        stride = pow(2, predict_level)

        test_patches_xy = slide.sample_all_patches(stride=stride)  # coordinates, ((0,0), (256, 0), (512, 0)...)
        x_indices = np.array([int(p[0] / stride) for p in test_patches_xy])  # [0, 1, 2, 3, ... 0, 1, 2, ...]
        y_indices = np.array([int(p[1] / stride) for p in test_patches_xy])  # [0, 0, 0, 0, ... 1, 1, 1, ...]

        level = min(slide.slide.level_count - 1, predict_level)  # some slides may not have level 8
        d = pow(2, (predict_level - level))
        shape = slide.slide.level_dimensions[level]
        shape = (int(shape[0] / d), int(shape[1] / d))
        logits = np.zeros((len(class_dict), shape[1] + 2, shape[0] + 2))  # avoid overflow

        # step 3: patch inference
        n_splits = 5
        test_patches_xy_split = np.array_split(test_patches_xy, n_splits)  # avoid index problem for too large data
        count = 0

        for i, split in enumerate(test_patches_xy_split):
            print(f"split {i + 1}/{n_splits} ({len(split)} patches) ...")

            slide.test_patch_coords = split
            test_ds = PatchDataset(slide)
            test_dl = DataLoader(test_ds, batch_size=batch_size,
                                 sampler=SubsetSequenceSampler(indices=list(range(len(split)))),
                                 num_workers=num_workers, pin_memory=pin_memory)

            preds = []
            model.eval()
            with torch.no_grad():
                for xb, _ in test_dl:
                    xb = xb.to(device)
                    if not is_listy(xb): xb = [xb]
                    out = model(*xb)
                    preds.append(to_detach(out, cpu=False))
            preds = to_float(torch.cat(preds))

            # step 4: stitch patch inference results into a map
            idxs_res = np.arange(len(split)) + count
            logits[:, y_indices[idxs_res],
            x_indices[idxs_res]] = preds.cpu().numpy().T  # res.shape=(n_sample, n_channel)
            count += len(split)
            torch.cuda.empty_cache()

        # step 5: store results
        logits = logits[:, :shape[1], :shape[0]]
        store = h5py.File(output_hdf5_filename, "a")
        store.create_dataset(slide.file.name + "_logits", data=logits, compression="gzip")
        store.close()

        # step 6: visualize
        if save_path:
            target = slide.get_full_annotation(level=level)[::d, ::d]
            thumbnail = np.array(slide.slide.get_thumbnail(size=slide.slide.level_dimensions[level]))[::d, ::d, :3]

            probs, pred_map = get_pred_map(output_hdf5_filename, slide, p=0.5, predict_level=predict_level,
                                           output_level=level)

            fig, axes = plt.subplots(1, 3, figsize=(15, 10))
            axes[0].imshow(thumbnail)
            im = axes[0].imshow(probs, cmap=plt.get_cmap('jet'), vmin=0, vmax=1, alpha=0.5)
            fig.colorbar(im, cax=make_axes_locatable(axes[0]).append_axes('right', size='5%', pad=0.05))
            axes[0].set_title("predicted tumor map")

            axes[1].imshow(thumbnail)
            im = axes[1].imshow(pred_map, cmap=ground_truth_color_map, vmin=0, vmax=1, alpha=0.5)
            fig.colorbar(im, cax=make_axes_locatable(axes[1]).append_axes('right', size='5%', pad=0.05))
            axes[1].set_title("predicted tumor map > 0.5")

            axes[2].imshow(thumbnail)
            im = axes[2].imshow(target, cmap=ground_truth_color_map, vmin=0, vmax=1, alpha=0.5)
            fig.colorbar(im, cax=make_axes_locatable(axes[2]).append_axes('right', size='5%', pad=0.05))
            axes[2].set_title("ground truth")

            plt.savefig(os.path.join(save_path, slide.file.name + ".png"))
            plt.close()

