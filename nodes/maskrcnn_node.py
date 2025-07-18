#!/usr/bin/env python3

import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import RegionOfInterest
from irsl_detection_msgs.msg import InstanceSegmentation
from irsl_detection_srvs.srv import GetInstanceSegmentation, GetInstanceSegmentationResponse

from irsl_instance_segmentation.inference_detectron2 import InferenceDetectron2


class InferenceDetectron2ROS:
    """ROS wrapper for Detectron2 instance segmentation inference.

    This class sets up a ROS service called `get_instance_segmentation`
    which takes a sensor_msgs/Image as input and returns instance segmentation
    results including class labels, bounding boxes, confidence scores, and masks.

    Attributes:
        bridge (CvBridge): Utility for converting between ROS and OpenCV images.
        model (InferenceDetectron2): Instance of the inference model.
        serv (rospy.Service): ROS service object.
    """

    def __init__(self):
        """Initializes the ROS node, service, and Detectron2 model."""
        self.bridge = CvBridge()
        self.model = InferenceDetectron2()

        # Start the service server with a descriptive service name
        self.serv = rospy.Service(
            'get_maskrcnn',
            GetInstanceSegmentation,
            self.inference_instancesegmentation
        )

    def inference_instancesegmentation(self, req: GetInstanceSegmentation._request_class) -> GetInstanceSegmentationResponse:
        """Handles instance segmentation requests.

        Args:
            req (GetInstanceSegmentationRequest): ROS service request containing an image.

        Returns:
            GetInstanceSegmentationResponse: Response message with detection results.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image_rgb = self.bridge.imgmsg_to_cv2(req.image, desired_encoding='rgb8')

            # Run inference
            results = self.model.inference(cv_image_rgb)

            # Prepare the response message
            ret = InstanceSegmentation()
            ret.header = req.image.header

            for result in results:
                # Populate class info
                ret.class_names.append(result["class"])
                ret.class_ids.append(result["class_id"])
                ret.scores.append(result["confidence"])

                # Populate bounding box
                box = RegionOfInterest()
                box.x_offset = int(result["bbox"][0])
                box.y_offset = int(result["bbox"][1])
                box.width = int(result["width"])
                box.height = int(result["height"])
                ret.boxes.append(box)

                # Convert and attach mask
                mask_np = np.array(result["mask"], dtype=np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_np, encoding="passthrough")
                mask_msg.header = req.image.header
                ret.masks.append(mask_msg)

            return GetInstanceSegmentationResponse(ret)

        except Exception as e:
            rospy.logerr(f"[InferenceDetectron2ROS] Inference failed: {e}")
            return GetInstanceSegmentationResponse()


if __name__ == '__main__':
    # Use a more descriptive and unique node name
    rospy.init_node('instance_segmentation_server')

    # Initialize service handler
    inference_ros = InferenceDetectron2ROS()

    # Keep the node alive
    rospy.spin()
