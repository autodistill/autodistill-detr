<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill DETR Module

This repository contains the code supporting the DETR base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[DETR](https://huggingface.co/facebook/detr-resnet-50) is a transformer-based computer vision model you can use for object detection. Autodistill supports use of the DETR Resnet 50 model developed by Meta Research.

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

# define an ontology to map class names to our DETR prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = DETR(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpg")
```

## License

The code in this repository is licensed under an .

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!