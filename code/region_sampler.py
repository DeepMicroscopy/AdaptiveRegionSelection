"""
Abstract class for region sampling.

Provides interface to sampling methods that allow same signature
"""

import abc

import numpy as np
import torch


def integral_helper(I, h, w):
    """Viola-Jones alg.
    calculate the sum of intensities over all windows of size (h,w)

    :param I: of size (H, W)
    :param h: window height
    :param w: window width
    :return:
    """
    integral = I.copy()
    for i in range(1, integral.shape[0]):
        integral[i, :] += integral[i - 1, :]
    for j in range(1, integral.shape[1]):
        integral[:, j] += integral[:, j - 1]

    integral = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]

    return integral  # of size (H-h, W-w)


def nms(x, y, width, height, score, n_query):
    """non-maximum suppression
    ref: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py

    :param x: x coordinates of the top left corner of each candidate
    :param y: y coordinates of the top left corner of each candidate
    :param width: width of each candidate (x direction), a scalar or a list with same length of x
    :param height: height of each candidate (y direction), a scalar or a list with same length of x
    :param score: priority scores for each candidate
    :param n_query: number of regions to select
    :return:
    """
    order = score.argsort()[::-1]

    keep = []
    while len(keep) < n_query and len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum((x + width)[i], (x + width)[order[1:]])
        yy2 = np.minimum((y + height)[i], (y + height)[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        inds = np.where(inter == 0.)[0]  # no intersection allowed
        order = order[inds + 1]

    return keep


class RegionSampler(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self.name = None

    def select_square_regions(self, priority_map, already_selected, n_query, region_size, grid=False):
        """

        :param priority_map: (C, H, W), note that negative values are skipped for accelerating computation
        :param already_selected:
        :param n_query:
        :param region_size:
        :param grid: if True, only consider non-overlapping region candidates
        :return: [[x11, y11, x12, y12], [x21, y21, x22, y22]...]
                 where (x11, y11) is the top left corner of the first region
                       (x12, y12) is the bottom right corner of the first region
                       (x21, y21) is the top left corner of the second region
                       (x22, y22) is the bottom right corner of the second region, etc.
        """
        # calculate the region priorities (i.e., the sum of the priorities of pixels within the region)
        I = integral_helper(priority_map, region_size, region_size)

        for b in already_selected:
            x1, y1, _, _ = b
            x1 = min(x1, I.shape[1] - 1)  # rounded error can appear when scaling between different levels
            y1 = min(y1, I.shape[0] - 1)
            I[y1, x1] = np.inf

        stride = region_size if grid else 1
        X, Y = np.meshgrid(np.arange(0, I.shape[1], stride), np.arange(0, I.shape[0], stride))
        I = I[Y, X]

        non_zero = np.where(I > 0)  # reduce values for sorting, save computation
        X, Y, priorities = X[non_zero], Y[non_zero], I[non_zero]

        selected = nms(X, Y, region_size, region_size, priorities, n_query + len(already_selected))
        selected = [[X[i], Y[i], X[i] + region_size, Y[i] + region_size] for i in selected[-n_query:]]

        return selected

    def select_non_square_regions(self, priority_map, already_selected, n_query, region_size):
        """consider a set of rectangles of the same size of region_size**2:
        1. rectangle width in the range of [region_size/2, region_size], step size being 256 pixel at level 0
        2. rectangle height in the range of [region_size/2, region_size], step size being 256 pixel at level 0

        The selection is in the same way as selecting square regions, just more candidates."""

        step = 2  # the step size for changing width/height, 2 pixel at select_level (i.e., 7) => 256 pixel at level 0
        priorities_list = []
        X_list = []
        Y_list = []
        width_list = []
        height_list = []

        for width in range(int(region_size / 2), region_size + 1, step):
            height = int(region_size * region_size / width)
            I = integral_helper(priority_map, h=height, w=width)

            for s in already_selected:
                x1, y1, x2, y2 = s
                if (x2 - x1) == width and (y2 - y1) == height:
                    x1 = min(x1, I.shape[1] - 1)  # rounded error can appear when scaling between different levels
                    y1 = min(y1, I.shape[0] - 1)
                    I[y1, x1] = np.inf

            X, Y = np.meshgrid(np.arange(0, I.shape[1]), np.arange(0, I.shape[0]))
            I = I[Y, X]

            non_zero = np.where(I > 0)  # reduce values for sorting, save computation
            X_list.extend(X[non_zero])
            Y_list.extend(Y[non_zero])
            priorities_list.extend(I[non_zero])
            width_list.extend(np.repeat(width, len(non_zero[0])))
            height_list.extend(np.repeat(height, len(non_zero[0])))

        for height in range(int(region_size / 2), region_size, step)[::-1]:
            width = int(region_size * region_size / height)
            I = integral_helper(priority_map, h=height, w=width)

            for s in already_selected:
                x1, y1, x2, y2 = s
                if (x2 - x1) == width and (y2 - y1) == height:
                    x1 = min(x1, I.shape[1] - 1)
                    y1 = min(y1, I.shape[0] - 1)
                    I[y1, x1] = np.inf

            X, Y = np.meshgrid(np.arange(0, I.shape[1]), np.arange(0, I.shape[0]))
            I = I[Y, X]

            non_zero = np.where(I > 0)  # reduce values for sorting, save computation
            X_list.extend(X[non_zero])
            Y_list.extend(Y[non_zero])
            priorities_list.extend(I[non_zero])
            width_list.extend(np.repeat(width, len(non_zero[0])))
            height_list.extend(np.repeat(height, len(non_zero[0])))

        selected = nms(np.array(X_list), np.array(Y_list), np.array(width_list), np.array(height_list),
                       np.array(priorities_list), n_query + len(already_selected))
        selected = [[X_list[i], Y_list[i], X_list[i] + width_list[i], Y_list[i] + height_list[i]] for i in
                    selected[-n_query:]]

        return selected
