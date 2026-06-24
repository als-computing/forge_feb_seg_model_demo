---
language:
- en # ISO language tag
tags:
- project: genesis # include on all GENESIS project models
- project: SYNAPS-I # include your _short_ model team name e.g. MOAT
- type: model # use other types include {agent, eval, framework, model, etc...}
- science: imaging # what kind of science is this for (e.g., materials, biology, lightsource, fusion, climate, etc.)
- risk: general # indicates level of risk review {general, reviewed, restricted}
license: {spdx_license_id} # use an SPDX license identifier https://spdx.org/licenses/
license_name: {license_name}  # If license = other (license not in https://hf.co/docs/hub/repositories-licenses), specify an id for it here, like `my-license-1.0`. if not delete this line
license_link: {license_link}  # If license = other, specify "LICENSE" or "LICENSE.md" to link to a file of that name inside the repo, or a URL to a remote file. if not delete this line
base_model: SAM3 # if fine tuning, include the basemodel url here
new_version:  # if this model has been superseeded by a new version, omit for now
datasets:
    - https://zenodo.org/records/19476729
    # a list of download URLs for dataset files used for training, mid-training, post-training, etc...
metrics:
  - AP
  - AP_50
  - AP_75
  - AP_small
  - AP_medium
  - AP_large
  - AR_maxDets=1
  - AR_maxDets=10
  - AR_maxDets=100
  - AR_small
  - AR_medium
  - AR_large
---

# SAM3 for Petiole Segmentation

This model is based on Meta AI's Segment Anything Model 3 (SAM3) and has been fine-tuned on a petiole micro-CT dataset acquired at the Advanced Light Source (ALS). It is designed for instance segmentation of anatomical structures in high-resolution synchrotron micro-CT images.

*Last Updated*: **2026-06-23**

## Developed by

Xiaoya Chong, Pradyumna Elavarthi, Tanny Chavez, Petrus Zwart, Alexander Hexemer

## Contributed by  

Andrew McElrone and Elizabeth Clark

## Model Changelog 
 
+ **2026-06-23** initial public version

## Model short description

+ Task: Segmentation with Concepts
+ Architecture: SAM3
+ Model Backbone: Perception Encoder
+ Parameters: 840M
+ Domain: Grapevine petiole micro-CT imaging
+ Use Case: Instance segmentation and concept-guided segmentation of anatomical structures in synchrotron micro-CT datasets

## Model description

1. This model is based on the SAM3 Perception Encoder (840M parameters) and has been fine-tuned for segmentation with concepts in grapevine petiole micro-CT images.
2. The model supports concept-guided instance segmentation, where users can provide a text prompt (e.g., "xylem vessels") to identify and segment individual instances of a specific cell type.
3. The training dataset consists of 42 manually annotated petiole micro-CT TIFF images with a resolution of 2560 × 2560 pixels and includes 7 cell classes. 
4. To decrease the number of objects in samples while preserving spatial resolution, each image was cropped into 640 × 640 patches with 25% overlap, resulting in more than 1,050 training images containing 107,282 annotations.
5. The model is intended for automated segmentation and analysis of plant structures in synchrotron micro-CT datasets.

## Finetuned from model (optional)

The base model is SAM3: https://github.com/facebookresearch/sam3

## Model Type

SAM3 Perception Encoder 
 
## Inputs and outputs

1. Input: The model accepts a 640 × 640 grayscale micro-CT image patch and a text prompt describing the target concept to segment (e.g., "xylem vessels", "cortex", or other annotated cell types). During training, high-resolution petiole micro-CT images (2560 × 2560 pixels) are divided into overlapping patches.
2. Input: To introduce scale variation during training, image patches are randomly resized and then padded to the target resolution required by the ViT backbone. This strategy improves robustness to objects of different sizes while maintaining compatibility with the fixed-resolution input constraint of the ViT architecture.
3. Output: The model outputs an instance segmentation mask of the same spatial dimensions as the input patch (640 × 640 pixels), together with confidence scores for the predicted instances. 
4. Output: For full-resolution inference on 2560 × 2560 images, the image is divided into overlapping patches and processed independently by SAM3. Predictions from overlapping regions are merged using a max-confidence strategy, where each pixel is assigned the label associated with the highest confidence score among overlapping predictions. The final output is a full-resolution segmentation mask (2560 × 2560 pixels) containing the segmented instances corresponding to the requested concept.

## Compute Infrastructure

The model is trained on NERSC with 4 GPUs.

### Hardware

This model was trained on 4 NVIDIA A100 GPUs on the NERSC Perlmutter supercomputer: https://www.nersc.gov/systems/perlmutter/

### Software

This model was trained using the SAM3 training framework and PyTorch 2.7.0 in a Conda environment with Python 3.12. The software environment can be reproduced using the following commands:
```
conda create -n sam3_demo python=3.12
conda activate sam3_demo
pip install .
```

The pyproject.toml file can be found at: https://github.com/als-computing/forge_feb_seg_model_demo/blob/main/pyproject.toml

## Papers and Scientific Outputs 

```bibtex
@unpublished{ChongEtAl2026_AIDrivenScanDiscovery,
  author    = {Chong, Xiaoya and Abramov, David and Elavarthi, Pradyumna and Clark, Elizabeth G. and Zwart, Petrus and Koepp, Wiebke and Parkinson, Dilworth Y. and McReynolds, Dylan and Bilieux, Hassina and Allan, Daniel and Yan, Hanfei and Weker, Johanna Nelson and Cherukara, Mathew Joseph and Mehta, Apurva and Schwarz, Nicholas and McElrone, Andrew and Chavez, Tanny and Hexemer, Alexander},
  date      = {2026},
  title     = {AI-Driven Scan-to-Discovery in 20 Minutes: Automated Segmentation of Synchrotron Micro-CT at Scale},
  note      = {Under review at Nature Communications},
  keywords  = {submitted},
}
```

## Model License

The model is currently not open source.

## Contact Info and Model Card Authors

Correspondence and requests for materials should be addressed to Tanny Chavez (tanchavez@lbl.gov) or Alexander Hexemer (ahexemer@lbl.gov).

# Intended Uses 

This section describes the use cases the model is intended for, including the languages and domains where it can be applied. Document areas that are out of scope or where performance may be limited.

## Intended Use

This model is intended for instance segmentation with concepts in synchrotron micro-CT images of grapevine petioles. Users can provide a text prompt corresponding to a biological structure or cell type (e.g., "xylem vessels", "cortex", or other annotated classes), and the model will identify and segment individual instances matching the specified concept. The model is designed for automated analysis of plant anatomical structures in high-resolution micro-CT datasets and is intended for research use.

### Primary Intended Users 

Beamline users at Beamline 8.3.2 of the Advanced Light Source (ALS). Scientists and facility users at ALS and other synchrotron or imaging facilities seeking automated concept-guided instance segmentation of high-resolution imaging datasets.

### Mission Relevance 

It is part of the SYNAPS-I project, which was supported by the U.S. DOE, Office of Science, Office of Advanced Scientific Computing Research (ASCR), award “The Transformational AI Models Consortium” under Contract No. AC02-05CH11231.

## Out-of-Scope Use Cases

This model is not intended for use on non-petiole micro-CT datasets without additional validation or fine-tuning.

# How to Use

This section is the most important for reusability of the model.

## Install Instructions

Install a conda environment by following the instructions at:
https://github.com/als-computing/forge_feb_seg_model_demo/tree/main#environment-1-sam3

## Training Configuration

To train the model, follow the official SAM3 training guide:
https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md

Prepare a [COCO-format](https://cocodataset.org/#format-results) dataset and a `.yaml` configuration file for your custom dataset, then finetune the model.

## Inference Configuration

To run inference, refer to the following notebook:
https://github.com/xiaoyachong/sam3/blob/finetune/examples/sam3_image_predictor_compare-plants-seg-640-25-42tiff.ipynb


# Code Snippets

## Training

```bash
python sam3/train/train.py \
    -c <your-yaml-file-path> \
    --use-cluster 0 \
    --num-gpus 1
```

## Inference

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

# Limitations

## Risks

> The most powerful AI systems may pose novel national security risks in the near future in areas
> such as cyberattacks and the development of chemical, biological, radiological, nuclear, or
> explosives (CBRNE) weapons, as well as novel security vulnerabilities. Because America
> currently leads on AI capabilities, the risks present in American frontier models are likely to be
> a preview for what foreign adversaries will possess in the near future. Understanding the
> nature of these risks as they emerge is vital for national defense and homeland security.

No direct cyberattack, CBRNE, or national-security enabling risks have been identified for this model. The model is a narrow computer vision system for segmenting grapevine petiole micro-CT images. Residual risks are limited to scientific and operational use: inaccurate segmentations may lead to incorrect measurements, performance may degrade outside the trained imaging/domain conditions, ambiguous prompts may produce incorrect masks, and users may over-rely on outputs without expert review. Standard data governance and software supply-chain controls should also be applied.

## Limitations

+ This model was trained exclusively on grapevine petiole micro-CT datasets acquired under specific imaging conditions and may not generalize well to other plant species, tissues, imaging modalities, or acquisition settings.

+ The model supports concept-guided segmentation only for the cell classes represented in the training dataset. Performance may degrade when prompted with unseen concepts or structures that were not included during training.

# Training details 

In this section you should describe all the relevant aspects of training that are useful from a reproducibility perspective. This includes any preprocessing and postprocessing that were done on the data, as well as details such as the number of epochs the model was trained for, the batch size, the learning rate, and so on. 

## Training data 

The model was trained on a manually annotated grapevine petiole micro-CT dataset acquired at Beamline 8.3.2 of the Advanced Light Source (ALS). The dataset was selected to support concept-guided instance segmentation of plant vascular structures and contains annotations for seven cell classes, including xylem vessels and other anatomically relevant tissue types.

The training data consist of 42 high-resolution micro-CT TIFF images with a spatial resolution of 2560 × 2560 pixels. To increase the number of training samples while preserving image resolution, each image was divided into overlapping 640 × 640 patches with 25% overlap, resulting in more than 1,050 training images and 107,282 instance annotations.

During training, image patches were randomly resized and padded to the target resolution required by the ViT backbone. This preprocessing strategy introduces scale variation while maintaining compatibility with the fixed-resolution input requirements of the SAM3 architecture.

Dataset DOI: https://doi.org/10.5281/zenodo.19476729

## Training Procedure 

The model was fine-tuned using the SAM3 training framework on the petiole micro-CT dataset described above. Training data preprocessing included patch extraction from high-resolution images, random resizing, and padding to the target input resolution required by the ViT backbone. During inference, overlapping patch predictions were merged using a max-confidence strategy to generate full-resolution segmentation masks.

All training hyperparameters and configuration settings, including the learning rate, batch size, optimizer, training schedule, data preprocessing pipeline, augmentation settings, and model architecture parameters, are specified in the training configuration file:

https://github.com/xiaoyachong/sam3/blob/finetune/sam3/train/configs/plants/train_patches_seg_640_25_42tiff_7classes.yaml

### Reproducibility Information (optional)

- Random seed used: 42  
- Training platform: NERSC Perlmutter
- GPU: NVIDIA A100 GPUs
- Software framework: SAM3 and PyTorch
- Training configuration: https://github.com/xiaoyachong/sam3/blob/finetune/sam3/train/configs/plants/train_patches_seg_640_25_42tiff_7classes.yaml

## Pre-training information 

This model is initialized from the pretrained Segment Anything Model 3 (SAM3) Perception Encoder (840M parameters) released by Meta AI. The SAM3 foundation model was pretrained by Meta on a large-scale image segmentation corpus using the SAM3 training pipeline. Details of the pretraining datasets, model architecture, optimization procedures, loss functions, hyperparameters, and training schedules are described in the official SAM3 publication (https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) and repository (https://github.com/facebookresearch/sam3).


# Evaluation details 

Provide an indication of how well the model performs on the evaluation dataset. If the model uses a decision threshold, either provide the decision threshold used in the evaluation, or provide details on evaluation at different thresholds for the intended uses. 

## Evaluation data  

The evaluation dataset was derived from the same manually annotated grapevine petiole micro-CT dataset used for training and is available at:

Dataset DOI: https://doi.org/10.5281/zenodo.19476729

A 20% hold-out subset of the dataset was reserved for model evaluation and was not used during training. The dataset consists of high-resolution synchrotron micro-CT images of grapevine petioles acquired at Beamline 8.3.2 of the Advanced Light Source (ALS), with annotations spanning seven cell classes.

## Evaluation Procedure

The model was evaluated on a held-out 20% subset of the grapevine petiole micro-CT dataset. Evaluation focused on **concept-guided instance segmentation**, where the model receives an image patch and a text prompt (e.g., *"xylem vessels"*) and predicts instance masks for the specified concept.

Performance was measured using the standard **COCO evaluation protocol**, including the following metrics:

* AP
* AP_50
* AP_75
* AP_small
* AP_medium
* AP_large
* AR_maxDets=1
* AR_maxDets=10
* AR_maxDets=100
* AR_small
* AR_medium
* AR_large

The reported metrics provide a benchmark for model performance on unseen petiole micro-CT data.

## Evaluation results 

The model achieved an AP of 54.8 and an AP_50 of 70.1 on the held-out evaluation dataset, demonstrating strong concept-guided instance segmentation performance. Performance on small cells was lower than on medium and large objects due to their small size, dense distribution, and ambiguous boundaries. Nevertheless, the model maintained robust performance across object scales, indicating its effectiveness in segmenting complex biological structures in petiole micro-CT images.
