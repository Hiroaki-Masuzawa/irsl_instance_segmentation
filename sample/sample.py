#!/usr/bin/env python3

import numpy as np
import cv2

import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import RegionOfInterest
from irsl_detection_msgs.msg import InstanceSegmentation
from irsl_detection_srvs.srv import (
    GetInstanceSegmentation,
    GetInstanceSegmentationResponse,
)


if __name__ == "__main__":
    bridge = CvBridge()
    image_cv = cv2.imread("input.jpg")
    image_msg = bridge.cv2_to_imgmsg(image_cv, encoding="bgr8")
    option_msg = String()
    srv_name = "get_maskrcnn"
    srv_name = "get_ram_grounding_sam"
    rospy.wait_for_service(srv_name)
    try:
        get_instance_segmentation = rospy.ServiceProxy(
            srv_name, GetInstanceSegmentation
        )
        resp = get_instance_segmentation(image_msg, option_msg)
        result = resp.result
        result_img = np.copy(image_cv)

        colors = [
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
            (255, 0, 255),
        ]
        for i, (name, score, box) in enumerate(
            zip(result.class_names, result.scores, result.boxes)
        ):
            cv2.rectangle(
                result_img,
                (box.x_offset, box.y_offset),
                (box.x_offset + box.width, box.y_offset + box.height),
                colors[i%len(colors)],
                2,
            )
            mask = bridge.imgmsg_to_cv2(result.masks[i], desired_encoding='passthrough')
            for v in range(result_img.shape[0]):
                for u in range(result_img.shape[1]):
                    if mask[v,u] != 0:
                        result_img[v,u] = (result_img[v,u]+np.array(colors[i%len(colors)]))//2
        cv2.imwrite("output.png", result_img)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
