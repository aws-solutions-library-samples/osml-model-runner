#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase

import numpy as np


class TestNMSMethods(TestCase):
    def setUp(self):
        # Setup mock bounding box data, scores, and labels
        self.boxes = [
            np.array([[0.1, 0.1, 0.4, 0.4], [0.15, 0.15, 0.45, 0.45], [0.6, 0.6, 0.9, 0.9]]),  # Model 1
            np.array([[0.2, 0.2, 0.5, 0.5], [0.7, 0.7, 1.0, 1.0]]),  # Model 2
        ]
        self.scores = [np.array([0.9, 0.85, 0.6]), np.array([0.8, 0.7])]
        self.labels = [np.array([1, 1, 2]), np.array([1, 2])]

    def test_prepare_boxes(self):
        from aws.osml.model_runner.common.ensemble_boxes_nms import prepare_boxes

        # Create boxes with invalid values (< 0 and > 1) and zero-area boxes
        invalid_boxes = np.array([[-0.1, 0.2, 1.1, 1.2], [0.5, 0.5, 0.5, 0.5]])
        invalid_scores = np.array([0.9, 0.8])
        invalid_labels = np.array([1, 1])

        filtered_boxes, filtered_scores, filtered_labels = prepare_boxes(invalid_boxes, invalid_scores, invalid_labels)

        # Check if invalid boxes are corrected and zero-area boxes are removed
        self.assertEqual(filtered_boxes.shape[0], 1)
        self.assertTrue(np.all(filtered_boxes >= 0) and np.all(filtered_boxes <= 1))

    def test_nms(self):
        from aws.osml.model_runner.common.ensemble_boxes_nms import nms

        # Test NMS with IoU threshold = 0.5
        final_boxes, final_scores, final_labels = nms(self.boxes, self.scores, self.labels, iou_thr=0.5)

        # Ensure NMS returned expected number of boxes
        self.assertEqual(final_boxes.shape[0], 4)

    def test_soft_nms(self):
        from aws.osml.model_runner.common.ensemble_boxes_nms import soft_nms

        # Test Soft-NMS with linear method
        final_boxes, final_scores, final_labels = soft_nms(self.boxes, self.scores, self.labels, method=1, iou_thr=0.5)

        # Ensure Soft-NMS returned expected number of boxes
        self.assertEqual(final_boxes.shape[0], 5)

    def test_cpu_soft_nms(self):
        from aws.osml.model_runner.common.ensemble_boxes_nms import cpu_soft_nms_float

        # Test the internal Soft-NMS function with linear method
        dets = np.array([[0.1, 0.1, 0.4, 0.4], [0.15, 0.15, 0.45, 0.45], [0.6, 0.6, 0.9, 0.9]])
        scores = np.array([0.9, 0.85, 0.6])

        keep = cpu_soft_nms_float(dets, scores, Nt=0.5, sigma=0.5, thresh=0.5, method=1)

        # Ensure Soft-NMS returns the correct number of boxes
        self.assertEqual(len(keep), 2)

    def test_nms_float_fast(self):
        from aws.osml.model_runner.common.ensemble_boxes_nms import nms_float_fast

        # Test the fast NMS implementation
        dets = np.array([[0.1, 0.1, 0.4, 0.4], [0.15, 0.15, 0.45, 0.45], [0.6, 0.6, 0.9, 0.9]])
        scores = np.array([0.9, 0.85, 0.6])

        keep = nms_float_fast(dets, scores, thresh=0.5)

        # Ensure fast NMS returns correct number of boxes
        self.assertEqual(len(keep), 2)

    def test_nms_float_fast_2(self):
        from aws.osml.model_runner.common.ensemble_boxes_nms import nms

        # Test the fast NMS implementation
        dets = np.array([[0.1, 0.1, 0.4, 0.4], [0.15, 0.15, 0.45, 0.45], [0.6, 0.6, 0.9, 0.9]])
        scores = np.array([0.9, 0.85, 0.6])
        weights = [0.1, 0.1, 0.1]

        keep = nms(self.boxes, self.scores, self.labels, iou_thr=0.5, weights=weights)
