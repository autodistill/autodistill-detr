<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?3"
      >
    </a>
  </p>
</div>

# Autodistill DETR Module

This repository contains the code supporting the DETR base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[DETR](https://huggingface.co/docs/transformers/model_doc/detr) is a transformer-based computer vision model you can use for object detection. Autodistill supports training a model using the Meta Research Resnet 50 checkpoint.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [DETR Autodistill documentation](https://autodistill.github.io/autodistill/base_models/detr/).

## Installation

To use DETR with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-detr
```

## Quickstart

```python
from autodistill_detr import DETR

# load the model
target_model = DETR()

# train for 10 epochs
target_model.train("./roads", epochs=10)

# run inference on an image
target_model.predict("./roads/valid/-3-_jpg.rf.bee113a09b22282980c289842aedfc4a.jpg")
```

## License

The code in this repository is licensed under an Apache 2.0 license. See the [Hugging Face model card for the DETR Resnet 50](https://huggingface.co/facebook/detr-resnet-50) model for more information on the model license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!