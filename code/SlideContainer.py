import cv2
import matplotlib.pyplot as plt
import openslide
import os.path
from skimage import measure
from staintools.miscellaneous.optical_density_conversion import convert_RGB_to_OD
import time
import xml.etree.ElementTree as ET
from utils import *


def make_dir(dataset_path, slide_num, datatype):
    """
    dataset structure:

    training:
        -- tumor
            - tumor_001.tif
        -- normal
            - normal_001.tif
        -- lesion_annotations
            -tumor_001.xml (no annotation for normal slides)
    testing
        -- images
            - test_001.tif
        -- lesion_annotations
            - test_001.xml (no annotation for normal slides)
    """
    if datatype == 'train_tumor_slide':
        path = dataset_path + "training/tumor/tumor_{}.tif".format(str(slide_num).zfill(3))

    elif datatype == 'train_tumor_slide_annotation':
        path = dataset_path + "training/lesion_annotations/tumor_{}.xml".format(str(slide_num).zfill(3))

    elif datatype == 'train_normal_slide':
        path = dataset_path + "training/normal/normal_{}.tif".format(str(slide_num).zfill(3))

    elif datatype == 'test_slide':
        path = dataset_path + "testing/images/test_{}.tif".format(str(slide_num).zfill(3))

    elif datatype == 'test_slide_annotation':
        path = dataset_path + "testing/lesion_annotations/test_{}.xml".format(str(slide_num).zfill(3))

    else:
        raise ValueError

    return path


def read_xml(annotation_xml):
    """
    read annotation from .xml file

    Arguments:
        annotation_xml: path of .xml file

    Return:
        a list of dict, with each including class label and contour coordinates

        [{'label': 1, 'vertices': [(36904.3, 66020.5), (36862.1, 66004.9), (36818.7, 65987.2)...]},
        ...]

    reference: https://github.com/baidu-research/NCRF/blob/master/wsi/data/annotation.py
    """

    """quote from README.md of dataset repository
    The provided XML files may have three groups of annotations ("_0", "_1", or "_2") which can be 
    accessed from the "PartOfGroup" attribute of the Annotation node in the XML file. Annotations 
    belonging to group "_0" and "_1" represent tumor areas and annotations within group "_2" are 
    non-tumor areas which have been cut-out from the original annotations in the first two groups.
    """

    root = ET.parse(annotation_xml).getroot()
    annotations_tumor = root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
    annotations_0 = root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
    annotations_1 = root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
    annotations_2 = root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
    annotations_positive = annotations_tumor + annotations_0 + annotations_1
    annotations_negative = annotations_2

    res = []
    for annotation in annotations_positive:
        X = list(map(lambda x: float(x.get('X')), annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda x: float(x.get('Y')), annotation.findall('./Coordinates/Coordinate')))
        res.append({'label': 1, 'vertices': list(zip(X, Y))})

    for annotation in annotations_negative:
        X = list(map(lambda x: float(x.get('X')), annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda x: float(x.get('Y')), annotation.findall('./Coordinates/Coordinate')))
        res.append({'label': 0, 'vertices': list(zip(X, Y))})

    return res


class SlideContainer:
    def __init__(self,
                 dataset_path,
                 slide_num,
                 datatype,
                 class_dict=None,
                 mode="others",
                 level: int = 0,
                 patch_size: int = 256,
                 AL_annotations=None,
                 stain_aug_p: float = 0.,
                 stain_vector_pool=None):

        """
        :param dataset_path:
        :param slide_num:
        :param datatype: {'train_tumor_slide', 'train_tumor_slide_annotation', 'train_normal_slide', 'test_slide', 'test_slide_annotation'}
        :param mode: {"training", "inference", "others"}
        :param level: magnification level, level 0 corresponds to the highest magnification, level 1 reduces the resolution to half in both image directions, and etc.
        :param patch_size: the training patch size
        :param AL_annotations: .json file, incrementally enriched annotations during the AL process
        :param stain_aug_p: probability of performing stain augmentation
        :param stain_vector_pool: for stain augmentation, transfer the current staining style to a randomly selected style in the pool that contains the staining style of all training slides
        """

        self.datatype = datatype
        assert mode in ["training", "inference", "others"]
        self.class_dict = class_dict
        self.patch_size = patch_size
        self.AL_annotations = AL_annotations
        self.stain_aug_p = stain_aug_p
        self.stain_vector_pool = stain_vector_pool

        self.file = Path(make_dir(dataset_path, slide_num, datatype))
        self.slide = openslide.open_slide(str(self.file))
        self.level = level
        self.preprocessing_level = 5  # e.g., visualize thumbnail
        self.count = 0

        # For active learning experiments, we only reveal the annotations (i.e. self.polygon) within the
        # selected/annotated regions (i.e. self.annotated_boxes).
        # For baseline experiments with full annotations, the annotated area is the entire WSI.
        self.polygons = []
        if datatype in ["train_tumor_slide", "test_slide"]:
            annotation_xml = make_dir(dataset_path, slide_num, datatype + "_annotation")
            if os.path.exists(annotation_xml):
                self.polygons = read_xml(annotation_xml)  # coordinates at level 0

        # boxes bounding tissue areas are obtained from preprocessing and manually checked afterward. During inference,
        # only patches within the tissue boxes are predicted, the rest is set to background without prediction.
        self.tissue_boxes = self.load_tissue_box()
        # selected annotation regions during AL, where training/validation patches are randomly positioned
        self.annotated_boxes = self.load_annotated_box()
        # selected annotation region with tumor, for a balanced sampling of positive and negative patches
        self.tumor_boxes = []

        if mode == "training":
            # for stain augmentation
            with open("code/stain_vectors.json", 'r') as f:
                stainVectorDicts = json.load(f)
                self.stain_vector = np.array(
                    [d["stain_matrix"] for d in stainVectorDicts if d["filename"] == self.file.name][0])  # 2x3
                self.stain_vector_inv = np.linalg.pinv(self.stain_vector.T)

            # for avoiding extraction of background training patches
            self.white, _ = self.get_foreground_ostu()

            # force uniform patch sampling: a larger annotated region is sampled more
            annotated_boxes_area = [(b[2] - b[0]) * (b[3] - b[1]) for b in self.annotated_boxes]
            self.annotated_boxes_sampling_rate = np.array(annotated_boxes_area) / sum(annotated_boxes_area)

            # find tumor boxes -> balance positive and negative patches
            if datatype == 'train_tumor_slide':
                self.tumor_boxes = self.get_tumor_bboxes()
                tumor_area = [(b[2] - b[0]) * (b[3] - b[1]) for b in self.tumor_boxes]
                self.tumor_sampling_rate = np.array(tumor_area) / sum(tumor_area)

        if mode == "inference":
            self.test_patch_coords = []

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self.level]

    def coordinate_convert(self, coords, level_source, level_target):
        """convert coordinates from level_source to level_target

        :param coords: either an int or a nested list of box coordinates (e.g., [[1,2,3,4], [2,3,4,5]])
        :param level_source:
        :param level_target:
        :return:
        """
        if level_source == level_target:
            return coords

        def helper(i):
            return int(i / self.slide.level_downsamples[level_target] * self.slide.level_downsamples[level_source])

        if isinstance(coords, list):
            return [[helper(i) for i in c] for c in coords]
        else:
            return helper(coords)

    def load_bbox_helper(self, filename, key, level):
        """helper for loading bounding boxes of specific regions (e.g., annotated regions/ tissue area)

        :param filename:
        :param key:
        :param level: the level of returned coordinates
        :return: a list of all boxes, each with coordinates [x1, y1, x2, y2]
        """
        res = []
        with open(filename) as f:
            data = json.load(f)
        for d in data:
            if d['file'] == self.file.name:
                res = d[key]
                break
        return self.coordinate_convert(res, level_source=0, level_target=level)

    def load_tissue_box(self, level=None):
        """we detected tissue regions and recorded their bounding boxes for each slide. The
        use of tissue boxes to extract training patches has two advantages: 1) patches can
        be extracted at random positions within the bbox, so that training patches can overlap
        each other to improve data diversity, and 2) background patches can be avoided.

        :param level: the level of returned coordinates
        :return: a list of all boxes, each with coordinates [x1, y1, x2, y2]
        """
        level = level if level else self.level
        res = self.load_bbox_helper('code/tissue_boxes.json', key='tissue_boxes', level=level)
        if not res:
            shape = self.slide.level_dimensions[level]
            res = [[0, 0, shape[0], shape[1]]]
        return res

    def load_annotated_box(self, level=None):
        level = level if level else self.level
        res = []
        if self.AL_annotations:
            res = self.load_bbox_helper(self.AL_annotations, key="selected_regions", level=level)
        return res

    def get_tumor_bboxes(self, level=None):
        level = level if level else self.level
        # get annotated mask -> detect tumor regions and their bounding boxes
        full_anno = self.get_full_annotation(level=self.preprocessing_level)

        # reveal only annotated tumor area
        anno_map = np.zeros_like(full_anno)
        for b in self.annotated_boxes:
            x1, y1, x2, y2 = \
            self.coordinate_convert([b], level_source=self.level, level_target=self.preprocessing_level)[0]
            anno_map[y1:y2, x1:x2] = 1
        full_anno *= anno_map

        _, cc_properties = get_connected_components(full_anno)
        res = [[p.bbox[1], p.bbox[0], p.bbox[3], p.bbox[2]] for p in cc_properties]
        res = self.coordinate_convert(res, level_source=self.preprocessing_level, level_target=level)

        return res

    def get_foreground_ostu(self, level=None):
        level = level if level else self.preprocessing_level
        thumbnail = np.array(self.slide.get_thumbnail(self.slide.level_dimensions[level]))
        blurred = cv2.GaussianBlur(cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        white, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground = np.array(blurred <= white).astype(np.int64)

        return white, foreground

    def get_full_annotation(self, scatter=False, level=None, thickness=-1):
        """

        :param scatter: return each polygon vertices in the annotation .xml file
        :param level:
        :param thickness:
        :return:
        """
        level = level if level else self.preprocessing_level
        shape = self.slide.level_dimensions[level]
        res = np.zeros((shape[1], shape[0]), dtype=np.int8)

        for poly in self.polygons:
            vertices = np.array(self.coordinate_convert(poly['vertices'], level_source=0, level_target=level))
            if not scatter:
                cv2.drawContours(res, [vertices.reshape((-1, 1, 2)).astype(np.int32)], -1, poly['label'], thickness)
            else:
                res[vertices[:, 1], vertices[:, 0]] = poly['label']

        return res

    def click_estimation(self, boxes=None):
        """calculate the number of clicks that are required for annotate the boxes

        :param boxes: nested list, coordinates at level 0
        :return:
        """
        # ref: ``MetaBox+: A new Region Based Active Learning Method for Semantic Segmentation using Priority Maps''
        c_p = 0  # polygon contour
        c_i = 0  # intersection between the contours and box borders
        c_b = 0  # four vertices of each box
        c_c = 0  # the number of class in the box
        tumor_pixels = 0  # tumor pixels

        boxes = self.coordinate_convert(boxes, level_source=0, level_target=self.preprocessing_level)
        full_anno = self.get_full_annotation(level=self.preprocessing_level)
        full_anno_scatter = self.get_full_annotation(scatter=True, level=self.preprocessing_level)

        if not boxes:  # full annotation
            anno_map = np.ones_like(full_anno)
        else:
            anno_map = np.zeros_like(full_anno)
            for b in boxes:
                x1, y1, x2, y2 = b
                anno_map[y1:y2, x1:x2] = 1
                b_anno = full_anno[y1:y2, x1:x2]

                # intersection clicks
                c_i += np.sum(b_anno[0, 1:-1] != b_anno[0, :-2])
                c_i += np.sum(b_anno[1:-1, 0] != b_anno[:-2, 0])
                c_i += np.sum(b_anno[-1, 1:-1] != b_anno[-1, :-2])
                c_i += np.sum(b_anno[1:-1, -1] != b_anno[:-2, -1])

                # bounding box clicks
                c_b += 4

        full_anno_scatter *= anno_map
        c_p += np.sum(full_anno_scatter)

        full_anno *= anno_map
        c_c += np.max(measure.label(full_anno, connectivity=2))
        tumor_pixels += np.sum(full_anno)

        return [int(c_p), int(c_i), int(c_b), int(c_c), int(tumor_pixels)]

    def get_patch_x(self, x: int = 0, y: int = 0, size=None, level=None):
        """extracting patches from the WSI

        :param x: top left corner x
        :param y:  top left corner y
        :param size: patch size
        :param level: level for the inputs (x, y, size) and the output patch
        :return:
        """
        size = size if size else (self.patch_size, self.patch_size)
        level = level if level else self.level

        x = self.coordinate_convert(x, level_source=level, level_target=0)
        y = self.coordinate_convert(y, level_source=level, level_target=0)
        rgb = np.array(self.slide.read_region(location=(x, y), level=level, size=size))[:, :, :3]

        # Convert RGBA PNG to RGB with PIL: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        # from PIL import Image as pilImage
        # rgba = self.slide.read_region(location=(x, y), level=level, size=size)
        # rgb = pilImage.new('RGBA', rgba.size, (255, 255, 255))
        # rgb = pilImage.alpha_composite(rgb, rgba)
        # rgb = rgb.convert('RGB')

        return rgb

    def get_patch_y(self, x: int = 0, y: int = 0, size=None, level=None):
        """mask annotation of a patch

        :param x: top left corner x
        :param y:  top left corner y
        :param size: patch size
        :param level: magnification level
        :param return_label: a patch is labeled as positive if its central quarter contains any tissue pixel
        :return:
        """
        size = size if size else (self.patch_size, self.patch_size)
        level = level if level else self.level
        patch_y = np.zeros((size[1], size[0]), dtype=np.uint8)

        for poly in self.polygons:
            vertices = np.array(self.coordinate_convert(poly['vertices'], level_source=0, level_target=level))
            vertices -= (x, y)
            cv2.drawContours(patch_y, [vertices.reshape((-1, 1, 2)).astype(np.int32)], -1, poly['label'], -1)

        label = 1 if (patch_y[int(size[1] / 4):int(size[1] * 3 / 4),
                      int(size[0] / 4):int(size[0] * 3 / 4)] == 1).any() else 0
        return patch_y, label

    def get_new_patch(self):
        """

        :param return_label: if true return patch label (int), return patch annotation mask otherwise
        :return:
        """

        def uniform_sampling(boxes, boxes_sampling_rate):
            patch_x, patch_y, patch_label = None, None, None  # patch (rgb), patch annotation mask, patch label (int)
            for _ in range(25):
                x1, y1, x2, y2 = random.choices(boxes, boxes_sampling_rate, k=1)[0]
                xmin = int(random.uniform(x1, x2 - self.patch_size))
                ymin = int(random.uniform(y1, y2 - self.patch_size))

                patch_x = self.get_patch_x(xmin, ymin)
                patch_y, patch_label = self.get_patch_y(xmin, ymin)
                # resample a patch if it contains less than 1% foreground
                if np.sum(cv2.cvtColor(patch_x, cv2.COLOR_RGB2GRAY) < self.white) >= 0.01 * self.patch_size ** 2:
                    # if np.sum(cv2.cvtColor(patch_x, cv2.COLOR_RGB2GRAY) < self.white) >= 0.1 * self.patch_size ** 2:  # increase tissue content threshold when training with normal slides
                    break
            return patch_x, patch_y, patch_label

        def get_new_tumor_patch():
            patch_x, patch_y, patch_label = None, None, None
            for _ in range(25):
                patch_x, patch_y, patch_label = uniform_sampling(self.tumor_boxes, self.tumor_sampling_rate)
                if patch_label == 1:
                    break
            return patch_x, patch_y, patch_label

        def get_new_normal_patch():
            patch_x, patch_y, patch_label = None, None, None
            for _ in range(25):
                patch_x, patch_y, patch_label = uniform_sampling(self.annotated_boxes,
                                                                 self.annotated_boxes_sampling_rate)
                if patch_label == 0:
                    break
            return patch_x, patch_y, patch_label

        if self.tumor_boxes:
            if np.random.rand() < 0.5:
                x, y, l = get_new_tumor_patch()
            else:
                x, y, l = get_new_normal_patch()
        else:
            x, y, l = uniform_sampling(self.annotated_boxes, self.annotated_boxes_sampling_rate)

        return x, y, l

    def sample_all_patches(self, stride):
        """sample all patches within the tissue boxes sequentially with a stride, e.g. for inference

        :param stride:
        :return: a list of coordinates at self.level
        """
        patch_list = []

        for b in self.tissue_boxes:
            for ymin in np.arange(max(0, b[1] - int(stride / 2)), np.ceil((b[3] - stride / 2) / stride) * stride,
                                  stride):
                for xmin in np.arange(max(0, b[0] - int(stride / 2)), np.ceil((b[2] - stride / 2) / stride) * stride,
                                      stride):
                    patch_list.append((int(xmin), int(ymin)))

        print(f"test patch sampling -- {self.file}: {len(patch_list)} patches")
        return patch_list

    def stain_normalization(self, patch, stain_vector_target):
        OD = convert_RGB_to_OD(patch).reshape((-1, 3))
        concentrations = (self.stain_vector_inv @ OD.T).T
        concentrations = np.maximum(concentrations, 1e-6)
        res = 255 * np.exp(-1 * concentrations @ stain_vector_target)
        return res.reshape(patch.shape).astype(np.uint8)

    def stain_augmentation(self, patch):
        stain_vector_target = random.choices(self.stain_vector_pool, k=1)[0]
        return self.stain_normalization(patch, stain_vector_target)


def show_selected_box_helper(boxes, ax, color):
    for s in boxes:
        x1, y1, x2, y2 = s
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color)
