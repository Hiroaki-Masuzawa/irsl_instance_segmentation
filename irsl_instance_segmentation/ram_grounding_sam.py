#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image
import re

from segment_anything import build_sam, build_sam_vit_b, build_sam_hq, SamPredictor
import GroundingDINO.groundingdino.datasets.transforms as T
from automatic_label_ram_demo import load_model
import torchvision.transforms as TS
from ram.models import ram
from ram import inference_ram
from automatic_label_ram_demo import get_grounding_output
import torchvision
import torch

# Import base class
from irsl_instance_segmentation.object_perception import ObjectPerception


class RAMGroundingSegmentAnything:
    def __init__(
        self,
        box_threshold=0.25,
        text_threshold=0.2,
        iou_threshold=0.5,
        device="cuda",
    ):
        config_file = "/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounded_checkpoint = "/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
        ram_checkpoint = "/Grounded-Segment-Anything/ram_swin_large_14m.pth"
        sam_checkpoint_h = "/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
        sam_checkpoint_b = "/Grounded-Segment-Anything/sam_vit_b_01ec64.pth"
        self.device = device

        self.ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit="swin_l")
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.device)
        self.grounded_model = load_model(
            config_file, grounded_checkpoint, device=self.device
        )
        # predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint_h).to(device))
        self.sam_predictor = SamPredictor(
            build_sam_vit_b(checkpoint=sam_checkpoint_b).to(self.device)
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold

        self.transform1 = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform2 = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])

    def cv2pil(self, image, ch_order="rgb"):
        """OpenCV型 -> PIL型"""
        new_image = image.copy()
        if ch_order == "rgb":
            if new_image.ndim == 2:  # モノクロ
                pass
            elif new_image.shape[2] == 3:  # カラー
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            elif new_image.shape[2] == 4:  # 透過
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)
        return new_image

    def inference_with_prompt(self, image, text_prompt = ""):
        image_pil = self.cv2pil(image, "rgb")
        image_rgb, _ = self.transform1(image_pil, None)

        if text_prompt == "":
            raw_image = image_pil.resize((384, 384))
            raw_image = self.transform2(raw_image).unsqueeze(0).to(self.device)
            res = inference_ram(raw_image, self.ram_model)
            tags = res[0].replace(" |", ",")
        else:
            tags = text_prompt

        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.grounded_model,
            image_rgb,
            tags,
            self.box_threshold,
            self.text_threshold,
            device=self.device,
        )

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        nms_idx = (
            torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        )
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        results = []
        if boxes_filt.shape[0] != 0:
            image_np = np.array(image_pil)  # RGB
            self.sam_predictor.set_image(image_np)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_filt, image_np.shape[:2]
            ).to(self.device)
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )
            value = 0
            mask_img = torch.zeros(masks.shape[-2:])
            for idx, mask in enumerate(masks):
                mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

            for phrase, bbox, mask in zip(pred_phrases, boxes_filt.cpu(), masks.cpu()):
                match = re.match(r"(.+?)\s*\(([^()]*)\)", phrase)
                if match:
                    class_name = match.group(1)
                    confidence = float(match.group(2))
                else:
                    class_name = phrase
                    confidence = 1.0
                x1, y1, x2, y2 = bbox.numpy().astype(np.float64)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = (x2 - x1) 
                height = (y2 - y1)

                contours, _ = cv2.findContours(
                    mask[0].numpy().astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                points = []
                for contour in contours:
                    for point in contour:
                        x, y = point[0]
                        points.append({"x": int(x), "y": int(y)})

                mask = mask.numpy().astype(np.int64)
                results.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "x": float(center_x),
                        "y": float(center_y),
                        "width": float(width),
                        "height": float(height),
                        "mask": mask[0].tolist(),
                        "points": points,
                    }
                )
        return results

    def inference(self, image):
        return self.inference_with_prompt(image, text_prompt="")
