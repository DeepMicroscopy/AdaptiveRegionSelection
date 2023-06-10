import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d
from shapely import geometry
from region_sampler import RegionSampler
from SlideContainer import *
from result_analysis import get_pred_map


def check_overlap(dets):
    """check if the first region overlaps with any other regions -> avoid selecting overlapping regions

    :param dets: 2d array, each row is the coordinates of a region [x1, y1, x2, y2]
    :return: bool

    reference: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    """
    if len(dets) == 1:
        return False

    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    xx1 = np.maximum(x1[0], x1[1:])
    yy1 = np.maximum(y1[0], y1[1:])
    xx2 = np.minimum(x2[0], x2[1:])
    yy2 = np.minimum(y2[0], y2[1:])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    return (inter > 0).any()


def random_region_sampling(slide, n_query, region_size, already_selected=None):
    print(f"random region selection ({slide.file.name}): {n_query} regions of size {region_size}"
          f" at level {slide.level}...")
    _, foreground = slide.get_foreground_ostu(slide.level)
    tissue_boxes = slide.tissue_boxes
    if not already_selected: already_selected = slide.annotated_boxes
    selected = []
    early_stopping = False

    trials = 0
    while len(selected) < n_query:
        trials += 1
        # early stopping for slides with very little tissue volume
        if trials >= 25:
            if early_stopping:
                print(f"too little tissue within the slide {slide.file.name}, random selection early stopped "
                      f"with {len(selected)} newly selected and {len(already_selected)} already selected regions")
                break
            else:  # expanding the area that could be selected
                tissue_boxes = [[max(0, b[0] - region_size),
                                 max(0, b[1] - region_size),
                                 min(b[2] + region_size, slide.slide_shape[0]),
                                 min(b[3] + region_size, slide.slide_shape[1])] for b in tissue_boxes]
                print(f"{slide.file.name} expanded...")
                early_stopping = True
                trials = 0

        x1, y1, x2, y2 = random.choice(tissue_boxes)
        x = min(int(random.uniform(x1, x2 - region_size)), slide.slide_shape[0] - region_size)
        y = min(int(random.uniform(y1, y2 - region_size)), slide.slide_shape[1] - region_size)

        # the selected region should contain at least 10% of tissue
        if np.sum(foreground[y:y + region_size, x:x + region_size]) >= 0.1 * region_size * region_size:
            candidate = [x, y, x + region_size, y + region_size]
            if not check_overlap([candidate] + selected + already_selected):
                selected.append(candidate)

    return selected


def standard_region_sampling(slide, priority_map, n_query, region_size, grid=False, squared=True):
    """region priority equals the sum of pixel priorities

    :param priority_map:
    :param slide:
    :param n_query:
    :param region_size:
    :param grid: if True, only consider non-overlapping region candidates
    :param squared:  if False, consider (non-square) rectangular region candidates
    :return:
    """
    print(f"standard region selection ({slide.file.name}, squared={squared}): {n_query} regions of size {region_size}"
          f" at level {slide.level}...")
    region_sampler = RegionSampler()
    if squared:
        selected = region_sampler.select_square_regions(priority_map, slide.annotated_boxes, n_query, region_size, grid)
    else:
        selected = region_sampler.select_non_square_regions(priority_map, slide.annotated_boxes, n_query, region_size)

    return selected


def adaptive_region_sampling(slide, priority_map, n_query, region_size, save_path=None):
    """select regions of variant shape and size that best fits the detected informative area

    :param slide:
    :param priority_map:
    :param n_query:
    :param region_size:
    :return:
    """
    print(f"adaptive region selection ({slide.file.name}): {n_query} regions of size {region_size}"
          f" at level {slide.level}...")

    # region size in the range [(0.5*region_size)**2, (1.5*region_size)**2] to avoid extreme selections
    l_max = int(region_size * 3 / 2)
    l_min = int(region_size / 2)

    priority_map = medfilt2d(priority_map)  # avoid outliers when selecting pixels with the highest priorities

    for s in slide.annotated_boxes:
        x1, y1, x2, y2 = s
        priority_map[y1:y2, x1:x2] = 0

    fig, axes = plt.subplots(1, n_query, figsize=(n_query * 5, 10))
    selected = []
    for n in range(n_query):
        ax = axes[n]
        x, y, w, h = 0, 0, 0, 0

        py, px = np.unravel_index(priority_map.argmax(), priority_map.shape)
        print(f"selecting {n + 1}/{n_query}: find pixel with the highest priority: ({px}, {py})")

        left = 98  # 98 percentile of values in the priority map
        right = 100

        for _ in range(10):  # 10 trails to find the adapted region, otherwise select a region centered at (y,x)
            p = (left + right) / 2
            res = priority_map >= np.percentile(priority_map, p)
            res = np.uint8(res * 255)
            ax.imshow(res, cmap="gray")

            # connected component detection and sort them from the largest to the smallest
            contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for (_, c) in
                        sorted(zip([cv2.contourArea(c) for c in contours], contours), key=lambda pair: pair[0])][::-1]

            for c in contours:
                # discard contours with less than 3 vertices
                if len(c) <= 3:
                    break

                # find the connected component that contains the highest priority pixel
                if geometry.Polygon(c.reshape((-1, 2))).contains(geometry.Point(px, py)):
                    # detect its bounding box
                    x, y, w, h = cv2.boundingRect(c)
                    print(f"threshold = {p}%, x = {x}, y = {y}, w = {w}, h = {h}")
                    break

            # make sure the selected region area is within the range, otherwise adjust the intensity threshold
            if w * h > l_max ** 2:
                left = p
            elif w * h < l_min ** 2:
                right = p
            else:
                break

        # when no bounding box within the size range detected in 10 trials since, e.g., the detected informative area
        # is too small, select the region centering at the highest priority pixel with minimum size
        if x == 0 and y == 0:
            w = l_min
            h = l_min
            x = min(max(0, int(px - int(l_min / 2))), priority_map.shape[1] - l_min - 1)
            y = min(max(0, int(py - int(l_min / 2))), priority_map.shape[0] - l_min - 1)

        # when the detected informative area is too small or too large and is not able to fall in the size range by
        # adjusting the intensity threshold with 10 trials, e.g., when the area of the detected connected component
        # is always larger than the upperbound of the size range, adjust its size
        if w * h < l_min ** 2 or w * h > l_max ** 2:
            new_w = min(max(w, l_min), l_max)
            x = min(max(0, int(x + w / 2 - new_w / 2)), priority_map.shape[1] - new_w - 1)
            w = new_w
            new_h = min(max(h, l_min), l_max)
            y = min(max(0, int(y + h / 2 - new_h / 2)), priority_map.shape[0] - new_h - 1)
            h = new_h

        print(f"final selection-----------------------------------------x = {x}, y = {y}, w = {w}, h = {h}")
        priority_map[y:y + h, x:x + w] = 0
        selected.append([x, y, x + w, y + h])  # at select_level
        show_selected_box_helper(slide.annotated_boxes + selected[:-1], ax, "green")  # already selected
        show_selected_box_helper(selected[-1:], ax, "red")  # newly selected

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, slide.file.name.split(".")[0] + "_adapt.png"))
    plt.close()
    return selected

def select_regions(sampling_strategy, dataset_path, slide_name_list, predict_level, select_level, region_size,
                   n_query, pred_h5py_file, AL_annotations, set_name, save_path):
    """select (n_query) regions of size (region_size) for each slide in the (slide_name_list)
    according to (sampling_strategy) at level (select_level), log the coordinates (at level 0)
    of the selected regions as well as the estimated annotation costs in (out_filenames), store
    the visualization of selected regions in (save_path)

    a log example in output_filename:
        {"file": slide.file.name,
         "selected_regions": [[50, 40, 100, 90], [1000, 140, 1200, 190]],     <-- [top left x, top left y, bottom right x, bottom right right]
         "selected_clicks": [int(c_p), int(c_i), int(c_b), int(c_c), int(anno_pixels)],
         "set_name": "train"})
    """
    print(f"selecting on {slide_name_list}...")

    for idx, slide_name in enumerate(slide_name_list):
        print(f"selecting {idx + 1}/{len(slide_name_list)}: {slide_name}...")
        slide = SlideContainer(dataset_path=dataset_path,
                               slide_num=slide_name.split('.')[0].split('_')[1],
                               datatype="train_" + slide_name.split('_')[0] + "_slide",
                               level=select_level,
                               AL_annotations=AL_annotations)

        plot_img_list = []
        plot_name_list = []

        # "full/random/uncertain_standard/uncertain_non_square/uncertain_adapt"
        if sampling_strategy == "full":
            selected = slide.tissue_boxes
        elif sampling_strategy == "random":
            selected = random_region_sampling(slide, n_query, region_size)
        elif sampling_strategy.__contains__("uncertainty"):
            assert pred_h5py_file is not None
            prob, _ = get_pred_map(pred_h5py_file, slide, 0, predict_level, output_level=select_level)
            tissue = slide.get_foreground_ostu(select_level)[1]
            prob *= tissue
            priority_map = 1 - 2 * abs(prob - 0.5)
            plot_img_list.append(prob)
            plot_name_list.append("probability map")
            plot_img_list.append(priority_map)
            plot_name_list.append("uncertainty map")

            if sampling_strategy.__contains__("standard"):
                selected = standard_region_sampling(slide, priority_map, n_query, region_size, squared=True)
            elif sampling_strategy.__contains__("non_square"):
                selected = standard_region_sampling(slide, priority_map, n_query, region_size, squared=False)
            elif sampling_strategy.__contains__("adapt"):
                selected = adaptive_region_sampling(slide, priority_map, n_query, region_size, save_path)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        fig, axes = plt.subplots(1, len(plot_img_list) + 2, figsize=(5 * (len(plot_img_list) + 2), 10))
        for col in range(len(plot_img_list)):
            ax = axes[col]
            im = ax.imshow(plot_img_list[col], cmap='jet', vmin=0, vmax=1)
            fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
            show_selected_box_helper(slide.annotated_boxes, ax, "green")
            show_selected_box_helper(selected, ax, "red")
            ax.set_title(plot_name_list[col])

        ax = axes[-2]
        im = ax.imshow(slide.slide.get_thumbnail(size=slide.slide.level_dimensions[slide.level]))
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        show_selected_box_helper(slide.annotated_boxes, ax, "green")
        show_selected_box_helper(selected, ax, "red")
        ax.set_title(slide.file.name)

        ax = axes[-1]
        ax.imshow(slide.slide.get_thumbnail(size=slide.slide.level_dimensions[slide.level]))
        im = ax.imshow(slide.get_full_annotation(level=slide.level), cmap=ground_truth_color_map, alpha=0.5)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        show_selected_box_helper(slide.annotated_boxes, ax, "green")
        show_selected_box_helper(selected, ax, "red")
        ax.set_title("ground truth")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{slide.file.name.split('.')[0]}_{sampling_strategy}.png"))
        plt.close()

        # ===================================================================================================
        # always write level 0 coordinates in json files
        selected = slide.annotated_boxes + selected
        selected = slide.coordinate_convert(selected, level_source=slide.level, level_target=0)
        selected_clicks = slide.click_estimation(selected)

        selected_dict = {"file": slide.file.name,
                         "selected_regions": selected,
                         "selected_clicks": selected_clicks,
                         "set_name": set_name}

        with open(AL_annotations, 'r') as f:
            data = json.load(f)

        if slide.annotated_boxes:
            for i, d in enumerate(data):
                if d["file"] == slide.file.name:
                    data[i] = selected_dict
                    break
        else:
            data.append(selected_dict)

        with open(AL_annotations, 'w') as f:
            json.dump(data, f)
