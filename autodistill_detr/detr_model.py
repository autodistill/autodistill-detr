import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from PIL import Image
from autodistill.detection import CaptionOntology, DetectionBaseModel
from transformers import DetrImageProcessor, DetrForObjectDetection

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

@dataclass
class DETR(DetectionBaseModel):
    ontology: CaptionOntology
    detr_model: model

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.detr_model = model

    def predict(self, input: str) -> sv.Detections:
        labels = self.ontology.prompts()

        image = Image.open(input)
            
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]

        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        boxes = results["boxes"].tolist()

        detections = sv.Detections(
            xyxy=np.array(boxes),
            class_id=np.array(labels),
            confidence=np.array(scores),
        )

        return detections