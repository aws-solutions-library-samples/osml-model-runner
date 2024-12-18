# Copyright 2024 Amazon.com, Inc. or its affiliates.

import unittest

import numpy as np
import pytest


class TestNMSMethods(unittest.TestCase):
    """
    Unit tests for the Non-Maximum Suppression (NMS) and Soft-NMS functions.
    """

    def setUp(self):
        """
        Sets up mock bounding boxes, scores, and labels for testing.
        """
        # Bounding boxes (x1, y1, x2, y2) and scores from two models
        self.boxes = [
            np.array([[0.1, 0.1, 0.4, 0.4], [0.15, 0.15, 0.45, 0.45], [0.6, 0.6, 0.9, 0.9]]),  # Model 1
            np.array([[0.2, 0.2, 0.5, 0.5], [0.7, 0.7, 1.0, 1.0]]),  # Model 2
        ]
        self.scores = [
            np.array([0.9, 0.85, 0.6]),  # Scores for Model 1
            np.array([0.8, 0.7]),  # Scores for Model 2
        ]
        self.labels = [
            np.array([1, 1, 2]),  # Labels for Model 1
            np.array([1, 2]),  # Labels for Model 2
        ]

    def test_prepare_boxes(self):
        """
        Test the prepare_boxes function to ensure it:
        1. Corrects invalid box coordinates.
        2. Removes boxes with zero area.
        """
        from aws.osml.model_runner.common.ensemble_boxes_nms import prepare_boxes

        # Create invalid boxes with out-of-bound coordinates and zero area
        invalid_boxes = np.array([[-0.1, 0.2, 1.1, 1.2], [0.5, 0.5, 0.5, 0.5]])
        invalid_scores = np.array([0.9, 0.8])
        invalid_labels = np.array([1, 1])

        filtered_boxes, filtered_scores, filtered_labels = prepare_boxes(invalid_boxes, invalid_scores, invalid_labels)

        # Assertions
        assert filtered_boxes.shape[0] == 1
        assert np.all(filtered_boxes >= 0) and np.all(filtered_boxes <= 1)

    def test_nms(self):
        """
        Test the standard NMS function to ensure it suppresses overlapping boxes
        based on an IoU threshold of 0.5.
        """
        from aws.osml.model_runner.common.ensemble_boxes_nms import nms

        final_boxes, final_scores, final_labels = nms(self.boxes, self.scores, self.labels, 0.5)

        # Assertions
        assert final_boxes.shape[0] == 4

    def test_soft_nms(self):
        """
        Test the Soft-NMS function with the linear method (method=1).
        """
        from aws.osml.model_runner.common.ensemble_boxes_nms import soft_nms

        final_boxes, final_scores, final_labels = soft_nms(self.boxes, self.scores, self.labels, 1, 0.5)

        # Assertions
        assert final_boxes.shape[0] == 5

    def test_nms_fast(self):
        """
        Test the optimized NMS implementation (nms_fast) for speed and correctness.
        """
        from aws.osml.model_runner.common.ensemble_boxes_nms import nms_fast

        dets = np.array([[0.1, 0.1, 0.4, 0.4], [0.15, 0.15, 0.45, 0.45], [0.6, 0.6, 0.9, 0.9]])
        scores = np.array([0.9, 0.85, 0.6])

        keep = nms_fast(dets, scores, 0.5)

        # Assertions
        assert len(keep) == 2

    def test_nms_with_weights(self):
        """
        Test the NMS function with model weights applied to scores.
        """
        from aws.osml.model_runner.common.ensemble_boxes_nms import nms

        weights = [0.5, 0.5]  # Apply equal weights to both models
        final_boxes, final_scores, final_labels = nms(self.boxes, self.scores, self.labels, 0.5, weights=weights)

        # Assertions
        assert final_boxes.shape[0] == 4
        assert np.all(final_scores <= 1.0)  # Scores should remain normalized

    def test_invalid_input_lengths(self):
        """
        Test that NMS raises a ValueError when input lengths are mismatched.
        """
        from aws.osml.model_runner.common.ensemble_boxes_nms import nms

        # Mismatched input: boxes have fewer entries than scores and labels
        invalid_boxes = [np.array([[0.1, 0.1, 0.4, 0.4]])]  # 1 box
        invalid_scores = [np.array([0.9, 0.8])]  # 2 scores
        invalid_labels = [np.array([1, 2])]  # 2 labels

        # Verify that a ValueError is raised with a clear message
        with pytest.raises(ValueError):
            nms(invalid_boxes, invalid_scores, invalid_labels, 0.5)


if __name__ == "__main__":
    unittest.main()
