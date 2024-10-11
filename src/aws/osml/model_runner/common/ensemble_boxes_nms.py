# Copyright 2024 Amazon.com, Inc. or its affiliates.

"""
Non-Maximum Suppression (NMS) and Soft-NMS implementation for bounding boxes.

This module implements standard NMS, linear Soft-NMS, and Gaussian Soft-NMS
for bounding boxes with normalized coordinates. It supports weighted scores
and multiple labels.

Original implementation inspired by:
- https://github.com/ZFTurbo/Weighted-Boxes-Fusion

Author: ZFTurbo (https://kaggle.com/zfturbo)
Refactored for internal use in OSML.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from numba import jit


def prepare_boxes(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare boxes by correcting invalid coordinates and removing boxes with zero area.

    :param boxes: Array of shape (N, 4) with box coordinates, [x1, y1, x2, y2], where all values are normalized [0, 1].
    :param scores: Array of shape (N, ) with confidence scores for each box.
    :param labels: Array of shape (N, ) with labels for each box.

    :return: Tuple containing the filtered and corrected boxes, scores, and labels.
    """
    result_boxes = boxes.copy()

    cond = result_boxes < 0
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print("Warning. Fixed {} boxes coordinates < 0".format(cond_sum))
        result_boxes[cond] = 0

    cond = result_boxes > 1
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print("Warning. Fixed {} boxes coordinates > 1. Check that your boxes was normalized at [0, 1]".format(cond_sum))
        result_boxes[cond] = 1

    boxes1 = result_boxes.copy()
    result_boxes[:, 0] = np.min(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 2] = np.max(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 1] = np.min(boxes1[:, [1, 3]], axis=1)
    result_boxes[:, 3] = np.max(boxes1[:, [1, 3]], axis=1)

    area = (result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1])
    cond = area == 0
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print("Warning. Removed {} boxes with zero area!".format(cond_sum))
        result_boxes = result_boxes[area > 0]
        scores = scores[area > 0]
        labels = labels[area > 0]

    return result_boxes, scores, labels


def cpu_soft_nms_float(dets: np.ndarray, sc: np.ndarray, nt: float, sigma: float, thresh: float, method: int) -> np.ndarray:
    """
    Based on: https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
    It's different from original soft-NMS because we have float coordinates on range [0; 1]

    :param dets: boxes format [x1, y1, x2, y2]
    :param sc: scores for boxes
    :param nt: required iou
    :param sigma: Sigma value for Gaussian soft-NMS.
    :param thresh: Score threshold to filter boxes.
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS

    :return: index of boxes to keep
    """

    # indexes concatenate boxes with the last column
    n = dets.shape[0]
    indexes = np.array([np.arange(n)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1, x1, y2, x2]
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    scores = sc
    areas = (x2 - x1) * (y2 - y1)

    for i in range(n):
        # intermediate parameters for later parameters exchange
        tbd = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != n - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tbd
            # tbd = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            # tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            # tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > nt] = weight[ovr > nt] - ovr[ovr > nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep


@jit(nopython=True)
def nms_fast(dets: np.ndarray, scores: np.ndarray, thresh: float) -> list[np.ndarray[Any, Any]]:
    """
    It's different from original nms because we have float coordinates on range [0; 1]

    :param dets: numpy array of boxes with shape: (N, 5). Order: x1, y1, x2, y2, score. All variables in range [0; 1]
    :param scores:  numpy array of scores
    :param thresh: IoU value for boxes

    :return: index of boxes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_method(
    boxes,
    scores,
    labels,
    method: int = 3,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    thresh: float = 0.001,
    weights: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform NMS on a list of boxes, scores, and labels from multiple models.

    :param boxes: list of boxes predictions from each model, each box is 4 numbers. It has 3 dimensions
    (models_number, model_preds, 4). Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1].
    :param scores: list of scores for each model.
    :param labels: list of labels for each model.
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS.
    :param iou_thr: IoU value for boxes to be a match.
    :param sigma: Sigma value for SoftNMS.
    :param thresh: threshold for boxes to keep (important for SoftNMS).
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model.

    :return: tuple of (boxes, scores, labels) after NMS.
    """

    # Validate input lengths
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(f"Input lengths must match: boxes={len(boxes)}, scores={len(scores)}, labels={len(labels)}")

    # If weights are specified
    if weights is not None:
        if len(boxes) != len(weights):
            print("Incorrect number of weights: {}. Must be: {}. Skip it".format(len(weights), len(boxes)))
        else:
            weights = np.array(weights)
            for i in range(len(weights)):
                scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()

    # Do the checks and skip empty predictions
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for i in range(len(boxes)):
        if len(boxes[i]) != len(scores[i]) or len(boxes[i]) != len(labels[i]):
            print(
                "Check length of boxes and scores and labels: {} {} {} at position: {}. Boxes are skipped!".format(
                    len(boxes[i]), len(scores[i]), len(labels[i]), i
                )
            )
            continue
        if len(boxes[i]) == 0:
            # print('Empty boxes!')
            continue
        filtered_boxes.append(boxes[i])
        filtered_scores.append(scores[i])
        filtered_labels.append(labels[i])

    # We concatenate everything
    boxes = np.concatenate(filtered_boxes)
    scores = np.concatenate(filtered_scores)
    labels = np.concatenate(filtered_labels)

    # Fix coordinates and removed zero area boxes
    boxes, scores, labels = prepare_boxes(boxes, scores, labels)

    # Run NMS independently for each label
    unique_labels = np.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    for label in unique_labels:
        condition = labels == label
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        labels_by_label = np.array([label] * len(boxes_by_label))

        if method != 3:
            keep = cpu_soft_nms_float(
                boxes_by_label.copy(), scores_by_label.copy(), nt=iou_thr, sigma=sigma, thresh=thresh, method=method
            )
        else:
            # Use faster function
            keep = nms_fast(boxes_by_label, scores_by_label, thresh=iou_thr)

        final_boxes.append(boxes_by_label[keep])
        final_scores.append(scores_by_label[keep])
        final_labels.append(labels_by_label[keep])
    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)

    return final_boxes, final_scores, final_labels


def nms(
    boxes: List[np.ndarray],
    scores: List[np.ndarray],
    labels: List[np.ndarray],
    iou_thr: float = 0.5,
    weights: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short call for standard NMS

    :param boxes: list of boxes predictions from each model, each box is 4 numbers. It has 3 dimensions (models_number,
    model_preds, 4). Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1].
    :param scores: list of scores for each model.
    :param labels: list of labels for each model.
    :param iou_thr: IoU threshold value for boxes.
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model.

    :return: Tuple containing the final boxes, scores, and labels after NMS.
    """
    return nms_method(boxes, scores, labels, method=3, iou_thr=iou_thr, weights=weights)


def soft_nms(
    boxes: List[np.ndarray],
    scores: List[np.ndarray],
    labels: List[np.ndarray],
    method: int = 2,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    thresh: float = 0.001,
    weights: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform soft-NMS on the given set of boxes for each label.

    :param boxes: list of boxes predictions from each model, each box is 4 numbers. It has 3 dimensions
    (models_number, model_preds, 4). Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1].
    :param scores: list of scores for each model.
    :param labels: list of labels for each model.
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS.
    :param iou_thr: IoU value for boxes to be a match.
    :param sigma: Sigma value for SoftNMS.
    :param thresh: threshold for boxes to keep (important for SoftNMS).
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model.

    :return: Tuple containing the final boxes, scores, and labels after soft-NMS.
    """
    return nms_method(boxes, scores, labels, method, iou_thr, sigma, thresh, weights)
