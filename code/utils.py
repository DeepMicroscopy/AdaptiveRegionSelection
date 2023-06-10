from fastai.vision import *
from matplotlib.colors import ListedColormap
import skimage
from torch.utils.data import Dataset

ground_truth_color_map = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 1)])


def get_connected_components(pred_map):
    cc_map = skimage.measure.label(pred_map, connectivity=2)  # connected_component
    cc_properties = skimage.measure.regionprops(cc_map)
    return cc_map, cc_properties


def print_AL_annotations(AL_annotations):
    print(AL_annotations)
    with open(AL_annotations, "r") as rf:
        res = json.load(rf)
        print("-" * 50, f"training slides", "-" * 50)
        for a in res:
            if a["set_name"] == "train":
                print(a)
        print("-" * 50, f"validation slides", "-" * 50)
        for a in res:
            if a["set_name"] == "valid":
                print(a)


def prepare_AL_annotations(AL_annotations, last_AL_annotations=None):
    """at the beginning of each AL cycle, copy the AL annotation from last cycle"""
    res = []
    if last_AL_annotations:
        print_AL_annotations(last_AL_annotations)
        with open(last_AL_annotations, "r") as rf:
            res = json.load(rf)

    with open(AL_annotations, "w") as wf:
        json.dump(res, wf)


def annotation_cost(AL_annotations):
    with open(AL_annotations, "r") as f:
        res = json.load(f)

    # 1. annotated area

    annotated_area = 0

    for anno in res:
        regions = np.vstack([anno["selected_regions"]])

        # calculate the union of overlapping regions
        inter = 0
        x1 = regions[:, 0]
        y1 = regions[:, 1]
        x2 = regions[:, 2]
        y2 = regions[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        total = np.sum(areas)
        order = areas.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter += np.sum(w * h)
            order = order[1:]

        annotated_area += (total - inter)

    # 2. clicks
    # ref: ``MetaBox+: A new Region Based Active Learning Method for Semantic Segmentation using Priority Maps''
    # [int(c_p), int(c_i), int(c_b), int(c_c), int(tumor_pixels)]
    # [  367023       14     1316     1882 17462263]  --> full annotation
    clicks = np.sum(np.vstack([anno["selected_clicks"] for anno in res]), axis=0)
    C_c, C_p = 1882, 367023  # full annotation
    cost_a = (clicks[0] + clicks[1] + clicks[3]) / (C_p + C_c) * 100  # cost_A = (c_p + c_i + c_c) / (C_p + C_c)

    # 3. tumor ratio
    C_tumor_pixels = 17462263  # full annotation
    tumor_ratio = clicks[-1] / C_tumor_pixels * 100

    return annotated_area, clicks, cost_a, tumor_ratio


class ClassificationSlideDataset(Dataset):
    """for training, sampling n_sample patches from each slide per epoch"""

    def __init__(self, slides, tfms, class_dict, is_valid=False):
        self.slides = slides
        self.tfms = tfms
        self.class_dict = class_dict
        self.c = len(class_dict)
        self.is_valid = is_valid

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        slide = self.slides[idx]
        x, _, l = slide.get_new_patch()

        if (not self.is_valid) and random.random() < 0.5:  # perform stain augmentation
            x = slide.stain_augmentation(x)

        x = Image(pil2tensor(x / 255., dtype=np.float32))
        if self.tfms:
            x = x.apply_tfms(self.tfms)

        return x, LongTensor([l])

    def get_state(self, **kwargs):
        "Return the minimal state for export."
        state = {'tfms': self.tfms}
        return {**state, **kwargs}

    def show_batch(self, x, y, save_path):
        """visualize input data

        :param x: Tensor (bs,C,H,W)
        :param y: Tensor (bs,1)
        :return:
        """
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(16):
            ax = axes[i // 4][i % 4]
            ax.imshow(x[i].permute(1, 2, 0))
            ax.set_title(self.class_dict[y[i].item()])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class SegmentationSlideDataset(Dataset):
    """for training, sampling n_sample patches from each slide per epoch"""

    def __init__(self, slides, tfms, class_dict, is_valid=False):
        self.slides = slides
        self.tfms = tfms
        self.tfms_y = None if tfms is None else list(filter(lambda t: getattr(t, 'use_on_y', True), listify(tfms)))
        self.class_dict = class_dict
        self.c = len(class_dict)
        self.is_valid = is_valid

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        slide = self.slides[idx]
        x, y, _ = slide.get_new_patch()

        if (not self.is_valid) and random.random() < 0.5:  # perform stain augmentation
            x = slide.stain_augmentation(x)

        x = Image(pil2tensor(x / 255., dtype=np.float32))
        y = ImageSegment(pil2tensor(y, dtype=np.float32))
        if self.tfms:
            x = x.apply_tfms(self.tfms)
        if self.tfms_y:
            y = y.apply_tfms(self.tfms_y, do_resolve=False)

        return x, y

    def get_state(self, **kwargs):
        "Return the minimal state for export."
        state = {'tfms': self.tfms, 'tfms_y': self.tfms_y}
        return {**state, **kwargs}

    def show_batch(self, x, y, save_path):
        """visualize input data

        :param x: Tensor (bs,C,H,W)
        :param y: Tensor (bs,1)
        :return:
        """
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(16):
            ax = axes[i // 4][i % 4]
            ax.imshow(x[i].permute(1, 2, 0))
            ax.imshow(y[i].permute(1, 2, 0), cmap=ground_truth_color_map, alpha=0.5, vmin=0, vmax=1)

            size = y[i].shape
            label = 1 if (y[i][0, int(size[2] / 4):int(size[2] * 3 / 4),
                          int(size[1] / 4):int(size[1] * 3 / 4)] == 1).any() else 0

            ax.set_title(self.class_dict[label])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class CountSamplingRateCallback(LearnerCallback):
    def __init__(self, learn: Learner):
        super().__init__(learn)

    def on_train_begin(self, **kwargs):
        self.sampling_rate_train = torch.zeros((self.learn.data.c,))
        self.sampling_rate_valid = torch.zeros((self.learn.data.c,))

    def on_batch_begin(self, **kwargs):
        if kwargs["train"]:
            for y in kwargs["last_target"]: self.sampling_rate_train[int(y)] += 1
        else:
            for y in kwargs["last_target"]: self.sampling_rate_valid[int(y)] += 1

    def on_train_end(self, **kwargs: Any) -> None:
        print(f"trained with {torch.sum(self.sampling_rate_train)} patches "
              f"({self.sampling_rate_train[1] / torch.sum(self.sampling_rate_train) * 100:.2f}% tumor)")
        print(f"validated with {torch.sum(self.sampling_rate_valid)} patches "
              f"({self.sampling_rate_valid[1] / torch.sum(self.sampling_rate_valid) * 100:.2f}% tumor)")


class PatchDataset(Dataset):
    """for inference, predict all patches (self.slide.test_patch_coords) in the slide"""

    def __init__(self, slide):
        self.slide = slide

    def __len__(self):
        return len(self.slide.test_patch_coords)

    def __getitem__(self, index):
        xmin, ymin = self.slide.test_patch_coords[index]
        x = self.slide.get_patch_x(xmin, ymin)
        x = pil2tensor(x / 255., dtype=np.float32)

        return x, LongTensor([0])  # inference requires no label, all take 0


class SubsetSequenceSampler(Sampler[int]):
    r"""Samples elements sequentially from a given list of indices.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)
