# -*- coding: utf-8 -*-
"""
Adapted from:

Evaluation code for the Camelyon16 challenge on cancer metastases detection
@author: Babak Ehteshami Bejnordi
https://github.com/computationalpathologygroup/Camelyon16/blob/master/Python/Evaluation_FROC.py
"""
import time
from tqdm import tqdm
from skimage import measure
from result_analysis import *
from SlideContainer import SlideContainer
from utils import *

reference = get_reference()  # slide wise ground truth


def computeConfidenceScore(prob, mask, pooling, weighted=True):
    """Computes the confidence score of a detected lesion (connected component)

    Args:
        prob:      the probability map
        mask:      pixels within the connected component are labeled as 1, others as 0
        pooling:   Computes avg./max. probability within the connected component
        weighted:  multiply with the max. probability in the slide

    reference: winner's method based on the supplementary material of challenge paper:
               Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases
               in Women With Breast Cancer. JAMA. 2017;318(22):2199–2210. DOI: 10.1001/jama.2017.14585

    Returns:

    """
    if pooling == "avg":
        res = np.sum(prob * mask) / np.sum(mask)
    elif pooling == "sum":
        res = np.sum(prob * mask)
    elif pooling == "max":
        res = np.max(prob * mask)
    else:
        raise NotImplementedError
    if weighted:
        res *= np.max(prob)
    return res


def computeEvaluationMask(slide, resolution, evaluation_level):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        evaluation_level: The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """
    # get ground truth
    target = slide.get_full_annotation(level=evaluation_level)
    # Euclidean distance from the closest foreground element to each background element
    distance = nd.distance_transform_edt(1 - target)
    # extend each annotated tumor area with 75µm, which is the equivalent size of 5 tumor cells
    Threshold = 75 / (resolution * pow(2, evaluation_level) * 2)
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)
    return evaluation_mask


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore, the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275 / (resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i + 1)
    return Isolated_Tumor_Cells


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image

    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made

    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections

        TP_probs:   A list containing the probabilities of the True positive detections

        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

        detection_summary:   A python dictionary object with keys that are the labels
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate].
        Lesions that are missed by the algorithm have an empty value.

        FP_summary:   A python dictionary object with keys that represent the
        false positive finding number and values that contain detection
        details [confidence score, X-coordinate, Y-coordinate].
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label + 1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    FP_counter = 0
    if (is_tumor):
        for i in range(0, len(Xcorr)):
            HittedLabel = evaluation_mask[int(Ycorr[i] / pow(2, level)), int(Xcorr[i] / pow(2, level))]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i] > TP_probs[HittedLabel - 1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel - 1] = Probs[i]
    else:
        for i in range(0, len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter += 1

    num_of_tumors = max_label - len(Isolated_Tumor_Cells)
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve

    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs) / float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs) / float(sum(FROC_data[3]))
    return total_FPs, total_sensitivity


def plotFROC(total_FPs, total_sensitivity, FROC_score, title, save_path):
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

        title: contain information about post-processing steps

    Returns:
        -
    """
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)

    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')
    plt.title(f"{title} (FROC={FROC_score:.3f})", fontsize=12)
    plt.savefig(os.path.join(save_path, title + ".png"))
    plt.close()


def FROC_analysis(dataset_path, test_hdf5_file, predict_level, p, pooling, save_path, log_file):
    key_list = get_key_list(test_hdf5_file)
    slide_list = [key.split(".")[0] for key in key_list]  # "test_001.tif_logits" -> "test_001"
    Log(f"{'=' * 55}", log_file)
    Log(f"calculating FROC: {len(slide_list)} test slides, threshold = {p:.1f}", log_file)

    EVALUATION_MASK_LEVEL = 5  # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243  # pixel resolution at level 0

    FROC_data = np.zeros((4, len(slide_list)), dtype=object)  # name, FPs, TPs, n_tumors
    FP_summary = np.zeros((2, len(slide_list)), dtype=object)  # name, detected FPs
    detection_summary = np.zeros((2, len(slide_list)), dtype=object)  # name, lesions that should be detected

    is_tumor_count = 0
    for caseNum, case in tqdm(enumerate(slide_list), desc="collecting prediction data..."):
        slide = SlideContainer(dataset_path=dataset_path,
                               slide_num=case.split('_')[1],
                               datatype="test_slide",
                               level=EVALUATION_MASK_LEVEL)

        # get prediction
        prob_map, pred_map = get_pred_map(test_hdf5_file, slide, p, predict_level, fill_hole=False, itc_remove=False)

        # get connected components --> get coordinates and confidence
        cc_map, cc_properties = get_connected_components(pred_map)
        Xcorr, Ycorr, Probs = [], [], []
        for i in range(0, len(cc_properties)):
            mask = cc_map == i + 1  # indices of cc_map starts with 1, with 0 being background
            Ycorr.append((cc_properties[i].centroid[0] + 0.5) * pow(2, predict_level))  # level 0
            Xcorr.append((cc_properties[i].centroid[1] + 0.5) * pow(2, predict_level))
            Probs.append(computeConfidenceScore(prob_map, mask, pooling, weighted=True))

        is_tumor = reference[case][0].lower() == 'tumor'
        if is_tumor:
            is_tumor_count += 1
            evaluation_mask = computeEvaluationMask(slide, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            evaluation_mask = 0
            ITC_labels = []

        FROC_data[0][caseNum] = case  # slide name, e.g., "test_001"
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case

        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], \
        FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels,
                                                     EVALUATION_MASK_LEVEL)
    # Compute FROC curve
    total_FPs, total_sensitivity = computeFROC(FROC_data)

    # Compute FROC score
    sensitivity_at_predefined_FPs = np.interp((0.25, 0.5, 1, 2, 4, 8), total_FPs[::-1], total_sensitivity[::-1])
    FROC_score = np.mean(sensitivity_at_predefined_FPs)
    title = pooling + "_pooling_threshold_" + str(p)
    Log(f"{title}, FROC= {FROC_score:.3f}, "
        f"sensitivity_at_predefined_FPs={list(np.around(sensitivity_at_predefined_FPs, 3))}", log_file)

    # plot FROC curve
    plotFROC(total_FPs, total_sensitivity, FROC_score, title, save_path)

    return [*sensitivity_at_predefined_FPs, FROC_score]
