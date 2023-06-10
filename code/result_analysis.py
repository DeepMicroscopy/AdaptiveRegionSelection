import h5py
from scipy import ndimage as nd
from sklearn.metrics import roc_auc_score
from SlideContainer import SlideContainer
from utils import *
from user_define import experiment_config, Log


def get_reference():
    """ground truth reference: a dict of ground truth labels, e.g. {'test_001': 'Tumor', 'IDC', 'Macro',
                                                                   'test_002': ...}
    """
    reference = {}
    with open(experiment_config().dataset_path + "/testing/reference.csv", 'r') as f:
        data = csv.reader(f)
        for row in data:
            reference.update({row[0]: row[1:]})  # e.g. 'test_001': 'Tumor', 'IDC', 'Macro'
    return reference


reference = get_reference()


def get_key_list(filename):
    store = h5py.File(filename, "r")
    key_list = list(store.keys())
    store.close()
    return key_list


def get_pred_map(inference_hdf5_file, slide, p, predict_level, output_level=None, fill_hole=True, itc_remove=True):
    """generate prediction map (at output_level) based on the results (at predict_level) in the inference_hdf5_file"""

    store = h5py.File(inference_hdf5_file, "r")
    logits = store[slide.file.name + '_logits'][:]
    store.close()

    probs = torch.softmax(torch.tensor(logits), dim=0).cpu().numpy()
    probs = probs[1]  # tumor map
    not_predicted = np.prod((logits == 0), axis=0)  # background patches were not predicted
    probs *= (1 - not_predicted)

    if output_level and predict_level != output_level:
        shape = slide.slide.level_dimensions[output_level]
        probs = skimage.transform.resize(probs, output_shape=(shape[1], shape[0]), mode='constant', cval=0)

    pred_map = (probs > p) * 1.
    if fill_hole: pred_map = nd.morphology.binary_fill_holes(pred_map)
    if itc_remove: pred_map = remove_itc(pred_map, predict_level)

    return probs, pred_map


def conf_mat(y_true, y_pred, N):
    """fast calculation of confusion matrix"""
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat([y, torch.zeros(N * N - len(y), dtype=torch.long)])
    y = y.reshape(N, N)
    return y


def remove_itc(pred_map, predict_level):
    """ remove all Isolated_Tumor_Cells

    Isolated_Tumor_Cells (ITC) are single tumor cells or a cluster of tumor cells not larger
    than 0.2 mm (200µm) or less than 200 cells. Each pixel is 0.243µm*0.243µm in level 0.
    Therefore, the major axis of an ITC object at level 5 should be less than 200µm/(2^5*0.243)
    = 35.36 pixels.

    refer to official evaluation code:
    https://github.com/computationalpathologygroup/Camelyon16/blob/master/Python/Evaluation_FROC.py
    """
    cc_map, cc_properties = get_connected_components(pred_map)

    ITC_threshold = int(200 / (0.243 * pow(2, predict_level)))
    res = np.ones_like(pred_map, dtype=np.float32)
    for i in range(len(cc_properties)):
        if cc_properties[i].major_axis_length > ITC_threshold:
            mask = cc_map == i + 1  # cc_map index starts form 1
            res *= (1 - mask)
    return 1 - res


def slide_confidence(prob, pred_map, predict_level, weighted=True):
    """
    macrometastases: tumor cell cluster diameter >= 2mm
    micrometastases: tumor cell cluster diameter >0.2 mm and < 2mm
    ITC (solitary tumor cells or tumor cell clusters with diameter <= 0.2mm or less than 200 cells)
    The largest available metastasis determines the slide-based diagnoses.
    """
    confidence = 0

    if np.sum(pred_map) > 0:
        cc_map, cc_properties = get_connected_components(pred_map)
        cc_areas = [p.area for p in cc_properties]
        max_cc = np.argmax(cc_areas)  # largest connected component

        ITC_threshold = int(200 / (0.243 * pow(2, predict_level)))
        if cc_properties[max_cc].major_axis_length > ITC_threshold:
            mask = cc_map == max_cc + 1
            confidence = np.max(prob * mask)

        if weighted:
            confidence *= np.max(prob)

    return confidence


def iou_helper(alg_metrics_dict, log_file, class_dict):
    """ calculate average metrics for different categories: all tumor slides
                                                            all slides containing only micrometastases
                                                            all slides containing macrometastases

    alg_metrics_dic: a list of slide metrics (["IoU", "Precision", "Recall", "Dice"])
                     e.g. {'test_001': [[IoU_class0,       IoU_class1],
                                        [Precision_class0, Precision_class1],
                                        [Recall_class0,    Recall_class1],
                                        [Dice_class0,      Dice_class1],
                           'test_002': ...}

    return: a nested list of slide-averaged metrics
            [[tumor_IoU,       macrometastases_IoU,       micrometastases_IoU],
             [tumor_Precision, macrometastases_Precision, micrometastases_Precision],
             [tumor_Recall,    macrometastases_Recall,    micrometastases_Recall],
             [tumor_Dice,      macrometastases_Dice,      micrometastases_Dice]]
    """
    metric_list = ["IoU", "Precision", "Recall", "Dice"]
    n_classes = len(list(alg_metrics_dict.values())[0][0])

    # for idx in range(4):  # we only calculated IoU, uncomment to calculate precision, recall, dice as well
    idx = 0
    tumor = []
    macro = []
    micro = []

    for k in alg_metrics_dict.keys():  # slide names
        value = alg_metrics_dict[k][idx]

        if reference[k][0].lower() == 'tumor':
            tumor.append(value)
            if reference[k][2].lower() == 'macro':
                macro.append(value)
            else:
                micro.append(value)

    tumor = np.array(tumor)
    macro = np.array(macro)
    micro = np.array(micro)

    # metrics for each class
    tumor = np.mean(tumor, axis=0)
    macro = np.mean(macro, axis=0)
    micro = np.mean(micro, axis=0)

    Log(f"{'-' * 55}", log_file)
    Log(f"{metric_list[idx]:<10}{'tumor':>15}{'macro_tumor':>15}{'micro_tumor':>15}", log_file)
    for c in range(n_classes):
        Log(f"{class_dict[c]:<10}{tumor[c]:15.3f}{macro[c]:15.3f}{micro[c]:15.3f}", log_file)

    # metrics averaging over all classes
    tumor = np.mean(tumor)
    macro = np.mean(macro)
    micro = np.mean(micro)
    Log(f"{'-' * 55}", log_file)
    Log(f"{'average':<10}{tumor:15.3f}{macro:15.3f}{micro:15.3f}", log_file)

    return [tumor, macro, micro]


def auc_helper(confidence_dict, log_file):
    """ calculate average AUCs for different categories: all slides (tumor + normal)
                                                         all tumor slides
                                                         all slides containing only micrometastases
                                                         all slides containing macrometastases

    confidence_dict: a list of slide confidences: {'test_001': confidence,
                                                   'test_002': ...}

    return: a list of slide-averaged AUCs: [all_AUC, tumor_AUC, macrometastases_AUC, micrometastases_AUC]

    all_AUC differentiates all tumor slides from normal slides
    macro_AUC differentiates slides with macrometastases from normal slides excluding slides with only micrometastases
    micro_AUC differentiates slides with only micrometastases from normal slides excluding slides with macrometastases

    """

    def bootstrapping_helper(y_true, y_score):
        y_true, y_score = np.array(y_true), np.array(y_score)
        auc = roc_auc_score(y_true, y_score)  # point estimation

        bootstrapping_auc = []
        for _ in range(1000):
            idx = random.choices(range(len(y_true)), k=len(y_true))  # with replacement
            bootstrapping_auc.append(roc_auc_score(y_true[idx], y_score[idx]))
        return auc, np.percentile(bootstrapping_auc, 2.5), np.percentile(bootstrapping_auc, 97.5)

    alg_all, alg_macro, alg_micro = [], [], []  # algorithm prediction
    gt_all, gt_macro, gt_micro = [], [], []  # ground truth

    for k in confidence_dict.keys():
        confidence = confidence_dict[k]

        if reference[k][0].lower() == 'normal':
            alg_all.append(confidence)
            alg_macro.append(confidence)
            alg_micro.append(confidence)
            gt_all.append(0)
            gt_macro.append(0)
            gt_micro.append(0)
        else:
            alg_all.append(confidence)
            gt_all.append(1)

            if reference[k][2].lower() == 'macro':
                alg_macro.append(confidence)
                gt_macro.append(1)
            else:
                alg_micro.append(confidence)
                gt_micro.append(1)

    Log(f"{'-' * 55}", log_file)
    Log(f"{'':<35} AUC (95% CI)", log_file)
    all, low, high = bootstrapping_helper(gt_all, alg_all)
    Log(f"{'all (tumor n=' + str(sum(gt_all)) + ' + normal n=' + str(len(gt_all) - sum(gt_all)) + ')':<35}{all:.4f} ({low:.3f}-{high:.3f})",
        log_file)
    macro, low, high = bootstrapping_helper(gt_macro, alg_macro)
    Log(f"{'macro-tumor (n=' + str(sum(gt_macro)) + ')':<35}{macro:.4f} ({low:.3f}-{high:.3f})", log_file)
    micro, low, high = bootstrapping_helper(gt_micro, alg_micro)
    Log(f"{'micro-tumor (n=' + str(sum(gt_micro)) + ')':<35}{micro:.4f} ({low:.3f}-{high:.3f})", log_file)

    return [all, macro, micro]


def calculate_IoU(dataset_path, slide_list, class_dict, test_hdf5_file, p, log_file, predict_level):
    Log(f"{'=' * 55}", log_file)
    Log(f"calculate_IoU: {len(slide_list)} test slides, threshold = {p:.1f}", log_file)

    alg_metrics_dict = {}  # store IoU for each slide

    for case in slide_list:
        slide = SlideContainer(dataset_path=dataset_path,
                               slide_num=case.split(".")[0].split("_")[1],  # "test_001.tif" -> "001"
                               datatype="test_slide",
                               class_dict=class_dict)

        # get ground truth
        level = min(slide.slide.level_count - 1, predict_level)  # some slides may not have predict_level 8
        d = 2 ** (predict_level - level)
        target = slide.get_full_annotation(level=level)[::d, ::d]  # downsample to predict_level

        # get prediction
        _, pred_map = get_pred_map(test_hdf5_file, slide, p, predict_level)  # at predict_level

        # segmentation metrics
        cm = conf_mat(y_true=target.flatten(), y_pred=pred_map.flatten(), N=len(class_dict)).numpy()
        tp = np.diagonal(cm)
        posPred = cm.sum(axis=0)
        posGt = cm.sum(axis=1)

        eps = 1e-4
        IoU = tp / (posPred + posGt - tp + eps)
        alg_metrics_dict.update({case.split(".")[0]: [IoU]})
        # precision = tp / (posPred + eps)
        # recall = tp / (posGt + eps)
        # dice = 2 * precision * recall / (precision + recall + eps)
        # alg_metrics_dict.update({case.split(".")[0]: [IoU, precision, recall, dice]})

    return iou_helper(alg_metrics_dict, log_file, class_dict)


def calculate_AUC(dataset_path, slide_list, class_dict, test_hdf5_file, p, log_file, predict_level):
    Log(f"{'=' * 55}", log_file)
    Log(f"calculate_AUC: {len(slide_list)} test slides, threshold = {p:.1f}", log_file)

    confidence_dict = {}  # store slide prediction confidence, for calculating auc

    for case in slide_list:
        slide = SlideContainer(dataset_path=dataset_path,
                               slide_num=case.split(".")[0].split("_")[1],  # "test_001.tif" -> "001"
                               datatype="test_slide",
                               class_dict=class_dict)

        # get prediction
        prob, pred_map = get_pred_map(test_hdf5_file, slide, p, predict_level)

        confidence = slide_confidence(prob, pred_map, predict_level, weighted=True)
        confidence_dict.update({case.split(".")[0]: confidence})

    return auc_helper(confidence_dict, log_file)
