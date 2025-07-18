#!/usr/bin/env python3

import numpy as np
import cv2
from typing import List, Dict

# Import Detectron2 modules
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

# Import base class
from irsl_instance_segmentation.object_perception import ObjectPerception


class InferenceDetectron2(ObjectPerception):
    """Object detection and instance segmentation using Detectron2.

    This class utilizes Detectron2's pre-trained Mask R-CNN model to perform
    object detection and segmentation on input images.

    Attributes:
        cfg (CfgNode): Detectron2 configuration object.
        predictor (DefaultPredictor): Predictor object for inference.
        metadata (MetadataCatalog): Metadata including class names.
        class_names (List[str]): List of class names from metadata.
    """

    def __init__(self, threshold: float = 0.5, model_name: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        """Initializes the Detectron2 model and configuration.

        Args:
            threshold (float): Score threshold for filtering low-confidence detections.
            model_name (str): Model config name from Detectron2 model zoo.
        """
        # Load and merge model config from the model zoo
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))

        # Set the score threshold for detections
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # Load pre-trained model weights
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

        # Create the predictor for inference
        self.predictor = DefaultPredictor(self.cfg)

        # Get metadata and class names
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.class_names = self.metadata.get("thing_classes", [])

    def inference(self, image: np.ndarray) -> List[Dict]:
        """Performs object detection and segmentation on the input image.

        Args:
            image (np.ndarray): Input image in RGB format (OpenCV style).

        Returns:
            List[Dict]: A list of dictionaries, each containing information
            about a detected object:
                - class (str): Name of the detected class.
                - class_id (int): Class index.
                - confidence (float): Confidence score.
                - bbox (List[float]): Bounding box [x1, y1, x2, y2].
                - x (float): Center X-coordinate of the bounding box.
                - y (float): Center Y-coordinate of the bounding box.
                - width (float): Width of the bounding box.
                - height (float): Height of the bounding box.
                - mask (List[List[int]]): Binary mask of the object.
                - points (List[Dict[str, int]]): List of contour points as dictionaries with 'x' and 'y'.
        """
        # Run inference using the predictor
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")  # Move results to CPU

        results = []

        # Iterate over each detected instance
        for i in range(len(instances)):
            # Extract bounding box
            box = instances.pred_boxes.tensor[i].numpy().tolist()
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Extract class and score
            confidence = float(instances.scores[i])
            class_id = int(instances.pred_classes[i])

            # Get class name using cached class names
            class_name = self.class_names[class_id] if self.class_names else str(class_id)

            # Get the segmentation mask and extract contours
            mask = instances.pred_masks[i].numpy().astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Flatten contours to a list of point dictionaries
            points = []
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    points.append({"x": int(x), "y": int(y)})

            # Append the result for this instance
            results.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height,
                    "mask": mask.tolist(),
                    "points": points,
                }
            )

        return results
