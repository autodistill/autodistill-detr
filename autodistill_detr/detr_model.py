import os
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import supervision as sv
import torch
import torchvision
from autodistill.detection import DetectionTargetModel
from torch.utils.data import DataLoader
from transformers import (AutoModelForObjectDetection, DetrForObjectDetection,
                          DetrImageProcessor)

HOME = os.path.expanduser("~")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor):
        annotation_file_path = os.path.join(
            image_directory_path, "_annotations.coco.json"
        )
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {"image_id": image_id, "annotations": annotations}
        encoding = self.image_processor(
            images=images, annotations=annotations, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


class Detr(pl.LightningModule):
    def __init__(
        self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader, id2label
    ):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="facebook/detr-resnet-50",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.model_train_dataloader = train_dataloader
        self.model_val_dataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

    def train_dataloader(self):
        return self.model_train_dataloader

    def val_dataloader(self):
        return self.model_val_dataloader


@dataclass
class DETR(DetectionTargetModel):
    detr_model: AutoModelForObjectDetection

    def __init__(self):
        pass

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        with torch.no_grad():
            # load image and predict
            image = cv2.imread(input)
            inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)

            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)

            results = image_processor.post_process_object_detection(
                outputs=outputs, threshold=confidence, target_sizes=target_sizes
            )[0]

            return sv.Detections.from_transformers(transformers_results=results)

    def train(self, dataset, epochs: int = 100):
        train_directory = os.path.join(dataset, "train")
        val_directory = os.path.join(dataset, "valid")

        train_dataset = CocoDetection(
            image_directory_path=train_directory,
            image_processor=image_processor,
        )

        val_dataset = CocoDetection(
            image_directory_path=val_directory,
            image_processor=image_processor,
        )

        labels = train_dataset.coco.cats

        id2label = {k: v["name"] for k, v in labels.items()}

        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
        )
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False
        )

        model = Detr(
            lr=1e-4,
            lr_backbone=1e-5,
            weight_decay=1e-4,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            id2label=id2label,
        )

        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if DEVICE == "cuda" else "cpu",
            max_epochs=epochs,
            gradient_clip_val=0.1,
            accumulate_grad_batches=8,
            log_every_n_steps=5,
        )

        trainer.fit(model)

        model.model.save_pretrained("detr_model")

        self.model = DetrForObjectDetection.from_pretrained("detr_model").to(DEVICE)
