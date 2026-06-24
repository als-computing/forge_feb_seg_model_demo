---
language:
- en # ISO language tag
tags:
- project: genesis # include on all GENESIS project models
- project: SYNAPS-I # include your _short_ model team name e.g. MOAT
- type: model # use other types include {agent, eval, framework, model, etc...}
- science: imaging # what kind of science is this for (e.g., materials, biology, lightsource, fusion, climate, etc.)
- risk:general # indicates level of risk review {general, reviewed, restricted}
license: {spdx_license_id} # use an SPDX license identifier https://spdx.org/licenses/
license_name: {license_name}  # If license = other (license not in https://hf.co/docs/hub/repositories-licenses), specify an id for it here, like `my-license-1.0`. if not delete this line
license_link: {license_link}  # If license = other, specify "LICENSE" or "LICENSE.md" to link to a file of that name inside the repo, or a URL to a remote file. if not delete this line
base_model: DINOv3 and Encoder-only Mask Transformer (EOMT) # if fine tuning, include the basemodel url here
new_version:  # if this model has been superseeded by a new version, omit for now
datasets:
    - https://zenodo.org/records/19476729
    # a list of download URLs for dataset files used for training, mid-training, post-training, etc...
metrics:
    - Mean Intersection over Union (mIoU) # list of metrics used to monitor and evaluate model performance during training and validation

---

# DINOv3-Seg for Petiole Segmentation

DINOv3-Seg is a semantic segmentation model for grapevine petiole micro-CT images. The model combines the pretrained Meta DINOv3 vision foundation model with an EOMT segmentation head to perform dense pixel-wise prediction of cellular structures. It was fine-tuned on a manually annotated petiole micro-CT dataset containing six cell classes and is designed for automated semantic segmentation of plant cells in high-resolution synchrotron micro-CT data.

*Last Updated*: **2026-06-23**

## Developed by

Xiaoya Chong, Pradyumna Elavarthi, Tanny Chavez, Petrus Zwart, Alexander Hexemer

## Contributed by  

Andrew McElrone and Elizabeth Clark

## Model Changelog 
 
+ **2026-06-23** initial public version

## Model short description

+ Task: Semantic Segmentation
+ Architecture: DINOv3-Seg
+ Backbone: DINOv3 ViT-L/16
+ Segmentation Head: EOMT Mask Module
+ Domain: Grapevine petiole micro-CT imaging
+ Use Case: Semantic segmentation of six cell classes in high-resolution synchrotron micro-CT datasets

## Model description

1. This model combines the DINOv3 ViT-L/16 backbone (300M parameters) with an EOMT mask module for semantic segmentation of grapevine petiole micro-CT images.
2. The training dataset consists of 42 manually annotated petiole micro-CT TIFF images with a resolution of 2560 × 2560 pixels and includes 6 cell classes.
3. Unlike the SAM3 model, the full-resolution images are used directly for training and inference without patch-based cropping.
The DINOv3 backbone is initialized from pretrained weights, and the EOMT mask module is trained to predict dense pixel-wise segmentation maps for the six annotated cell classes.
4. The model is intended for automated semantic segmentation and analysis of plant structures in high-resolution synchrotron micro-CT datasets.

## Finetuned from model (optional)

The base model is DINOv3 ViT-L/16: https://github.com/facebookresearch/dinov3

## Model Type

DINOv3 ViT-L/16
 
## Inputs and outputs

1. Input: The model accepts a 2560 × 2560 grayscale micro-CT image as input. The training and inference pipelines operate directly on the full-resolution images without patch extraction or cropping.
2. Output: The model outputs a semantic segmentation mask with the same spatial dimensions as the input image (2560 × 2560 pixels), where each pixel is assigned one of the six annotated cell classes.


## Compute Infrastructure

The model is trained on NERSC with 4 GPUs.

### Hardware

This model was trained on 4 NVIDIA A100 GPUs on the NERSC Perlmutter supercomputer: https://www.nersc.gov/systems/perlmutter/

### Software

This model was trained using the DINOv3-Seg framework with PyTorch 2.9.1 in a Conda environment with Python 3.12. The software environment can be reproduced using the following commands:
```
conda create -n dino_demo python=3.12 
conda activate dino_demo 
pip install -r requirements.txt
```

The complete dependency specification can be found at:

https://github.com/als-computing/forge_feb_seg_model_demo/blob/main/requirements.txt

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

This model is intended for semantic segmentation of synchrotron micro-CT images of grapevine petioles. Given a high-resolution micro-CT image, the model predicts a pixel-wise segmentation map containing six annotated cell classes, enabling automated identification and analysis of plant structures. The model is designed for high-throughput analysis of plant tissues in synchrotron micro-CT datasets and is intended for research use.

### Primary Intended Users 

Beamline users at Beamline 8.3.2 of the Advanced Light Source (ALS). Scientists and facility users at ALS and other synchrotron or imaging facilities seeking automated semantic segmentation of high-resolution imaging datasets.

### Mission Relevance 

It is part of the SYNAPS-I project, which was supported by the U.S. DOE, Office of Science, Office of Advanced Scientific Computing Research (ASCR), award “The Transformational AI Models Consortium” under Contract No. AC02-05CH11231.

## Out-of-Scope Use Cases

This model is not intended for use on non-petiole micro-CT datasets without additional validation or fine-tuning.

# How to use 

This section is the most important for reusability of the model.  This section should include: 

## Install Instructions 

Install a conda environment by following the instructions at:
https://github.com/als-computing/forge_feb_seg_model_demo#environment-2-dino-and-cellpose

## Training configuration 

To train the model, we use the LightlyTrain framework, which provides an efficient workflow for fine-tuning DINOv3 with a semantic segmentation mask module. The model was fine-tuned following the official LightlyTrain semantic segmentation guide:

https://docs.lightly.ai/train/stable/semantic_segmentation.html

## Inference configuration 

To run inference, refer to the following notebook:
https://github.com/xiaoyachong/lightly_6classes/blob/main/inference-petioles.ipynb

# Code Snippets

## Training

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out/my_experiment",
        model="dinov3/vitl16-eomt",
        data={
            "train": {
                "images": "my_data_dir/train/images",   # Path to training images
                "masks": "my_data_dir/train/masks",     # Path to training masks
            },
            "val": {
                "images": "my_data_dir/val/images",     # Path to validation images
                "masks": "my_data_dir/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset
                0: "cortex",
                1: "pholem",
                2: "xylem vessels",
                # ...
            },
            # Optional: classes to ignore during training
            "ignore_classes": [0],
        },
    )
```

## Inference

```python
import lightly_train

model = lightly_train.load_model("out/my_experiment/exported_models/exported_best.pt")
masks = model.predict("path/to/image.jpg")
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
+ The model performs semantic segmentation for the six cell classes represented in the training dataset. Performance may degrade when applied to images containing unseen cell types, anatomical structures, or imaging characteristics not represented during training.

## Training data 

The model was trained on a manually annotated grapevine petiole micro-CT dataset acquired at Beamline 8.3.2 of the Advanced Light Source (ALS). The dataset was selected to support semantic segmentation of plant anatomical structures and contains annotations for six cell classes.

The training data consist of **42 high-resolution micro-CT TIFF images** with a spatial resolution of **2560 × 2560 pixels**. The dataset is diverse, including images acquired with both **4× and 10× magnification objectives**, as well as samples representing different developmental stages (**young and mature**) and hydration conditions (**hydrated and dehydrated**). This diversity helps improve model robustness across varying imaging conditions and biological states.

Dataset DOI: https://doi.org/10.5281/zenodo.19476729

## Training Procedure 

The model was fine-tuned using the LightlyTrain framework on the grapevine petiole micro-CT dataset described above. Full-resolution images (2560 × 2560 pixels) were used directly for training without image cropping or patch extraction.

All training hyperparameters and configuration settings, including the learning rate, batch size, optimizer, training schedule, data preprocessing pipeline, augmentation settings, and model architecture parameters, follow the default settings provided by the LightlyTrain semantic segmentation framework. Training was performed using the DINOv3 ViT-L/16 backbone with the EOMT mask module for semantic segmentation.

### Reproducibility Information (optional)

- Random seed used: 42  
- Training platform: NERSC Perlmutter
- GPU: NVIDIA A100 GPUs
- Software framework: LightlyTrain
- Training configuration: https://github.com/lightly-ai/lightly

## Pre-training information 

This model is initialized from the pretrained DINOv3 ViT-L/16 (300M parameters) foundation model released by Meta AI. DINOv3 was pretrained using large-scale self-supervised learning on diverse image datasets to learn general visual representations. Details of the pretraining datasets, optimization procedures, loss functions, training schedules, and hyperparameters are available in the official DINOv3 publication (https://arxiv.org/abs/2508.10104) and repository (https://github.com/facebookresearch/dinov3).

The semantic segmentation component is based on the Encoder-only Mask Transformer (EoMT) framework, which trains a lightweight mask module on top of a pretrained ViT encoder. The EoMT framework was originally developed using DINOv2-pretrained encoders and pre-trained on the Cityscapes and ADE20K semantic segmentation datasets.

# Evaluation details 

Provide an indication of how well the model performs on the evaluation dataset. If the model uses a decision threshold, either provide the decision threshold used in the evaluation, or provide details on evaluation at different thresholds for the intended uses. 

## Evaluation data  

The evaluation dataset was derived from the same manually annotated grapevine petiole micro-CT dataset used for training and is available at:

Dataset DOI: https://doi.org/10.5281/zenodo.19476729

A 20% hold-out subset of the dataset was reserved for model evaluation and was not used during training. The dataset consists of high-resolution synchrotron micro-CT images of grapevine petioles acquired at Beamline 8.3.2 of the Advanced Light Source (ALS), with annotations spanning seven cell classes.

## Evaluation Procedure

The model was evaluated on a held-out 20% subset of the grapevine petiole micro-CT dataset. Evaluation focused on semantic segmentation, where the model predicts a pixel-wise segmentation mask for six annotated cell classes from a full-resolution micro-CT image.

Performance was measured using mean Intersection over Union (mIoU), a standard metric for semantic segmentation that quantifies the overlap between predicted and ground-truth segmentation masks across all classes.

The reported mIoU provides a benchmark for model performance on unseen petiole micro-CT data.

## Evaluation results 

The model achieved an mIoU of 80.9 on the held-out evaluation dataset, demonstrating strong semantic segmentation performance on grapevine petiole micro-CT images. Initializing the model from an EoMT checkpoint that had been previously fine-tuned on the Cityscapes benchmark further improved performance, achieving an mIoU of 81.9 after fine-tuning on the petiole micro-CT dataset. 

