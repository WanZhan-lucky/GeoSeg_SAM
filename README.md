# GeoSeg-SAM

GeoSeg-SAM is a SAM-based adaptation framework for closed-set semantic segmentation of geological environments in remote sensing imagery. This repository provides the training code and configuration files for the WLK and YJS datasets.

## Repository Structure

```text
GeoSeg_SAM/
├── configs/
│   ├── wlk-256input.yaml
│   └── yjs-256input.yaml
├── datasets/
├── models/
├── requirements.txt
├── train.py
├── pretrained/                  # create this directory for the SAM checkpoint
│   └── sam_vit_b_01ec64.pth
├── wdatas/                      # create this directory for the datasets
│   ├── wlk-256data/
│   └── yjs-256data/
└── outputs/                     # generated during training
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/WanZhan-lucky/GeoSeg_SAM.git
cd GeoSeg_SAM
```

A CUDA-enabled PyTorch environment is recommended. One compatible setup is:

```bash
conda create -n geoseg-sam python=3.8 -y
conda activate geoseg-sam

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install seaborn prettytable
```

PyTorch and TorchVision may also be installed with versions appropriate for the local CUDA environment.

## SAM Pretrained Checkpoint

Download the official SAM ViT-B checkpoint:

- `sam_vit_b_01ec64.pth`: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Place it at:

```text
GeoSeg_SAM/
└── pretrained/
    └── sam_vit_b_01ec64.pth
```

The provided configuration files expect the checkpoint path:

```yaml
sam_checkpoint: ./pretrained/sam_vit_b_01ec64.pth
```

## Datasets

The processed WLK and YJS datasets used by this repository are publicly available from Google Drive:

- **WLK dataset**: https://drive.google.com/file/d/1qddYp5atQpAWB4UPek57OOoMwWakU7xp/view?usp=sharing
- **YJS dataset**: https://drive.google.com/file/d/12ZwTNiTJikjXY0O9xO9O1YUs4gWh0wj0/view?usp=sharing

After downloading and extracting the datasets, arrange them as follows:

```text
GeoSeg_SAM/
└── wdatas/
    ├── wlk-256data/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   └── val/
    │       ├── images/
    │       └── labels/
    └── yjs-256data/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/
```

The provided configurations use 256 × 256 inputs. The WLK configuration contains 13 classes, while the YJS configuration contains 8 classes.

## Training

### Train on the WLK dataset

```bash
python train.py \
  --config configs/wlk-256input.yaml \
  --path ./outputs \
  --name geoseg_wlk
```

### Train on the YJS dataset

```bash
python train_singlew_loadfrom.py \
  --config configs/yjs-256input.yaml \
  --path ./outputs \
  --name geoseg_yjs
```

The training script supports the following command-line arguments:

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file. |
| `--path` | Root directory for training outputs. |
| `--name` | Name of the experiment subdirectory. |
| `--tag` | Optional suffix appended to the experiment name. |

For example:

```bash
python train_singlew_loadfrom.py \
  --config configs/wlk-256input.yaml \
  --path ./outputs \
  --name geoseg_wlk \
  --tag run1
```

This creates an experiment directory named `geoseg_wlk_run1` under `./outputs/`.

## Output Files

For an experiment named `geoseg_wlk`, outputs are saved under:

```text
outputs/
├── geoseg_wlk/
│   ├── config.yaml
│   ├── model_epoch_last.pth
│   ├── model_epoch_best.pth
│   └── model_epoch_best_mIoU_*.pth   # saved when the corresponding condition is met
└── heatmap/
    └── heatmap_*.jpg
```

During training, the script evaluates segmentation performance using mIoU and saves the best-performing checkpoint according to the validation result.

## Configuration

The main training settings are controlled by the YAML files in `configs/`. Important fields include:

```yaml
train_dataset:
  dataset:
    args:
      root_path_1: ./wdatas/.../train/images
      root_path_2: ./wdatas/.../train/labels

val_dataset:
  dataset:
    args:
      root_path_1: ./wdatas/.../val/images
      root_path_2: ./wdatas/.../val/labels

sam_checkpoint: ./pretrained/sam_vit_b_01ec64.pth

model:
  args:
    num_classes: 13
    inp_size: 256

optimizer:
  name: adamw
  args:
    lr: 0.0002

epoch_max: 120
```

To use a different dataset, update the image and label paths, class list, `num_classes`, and other task-specific settings accordingly.

## Reproducibility Notes

- The default dataset paths are defined relative to the repository root under `./wdatas/`.
- The default SAM checkpoint path is `./pretrained/sam_vit_b_01ec64.pth`.
- It is recommended to explicitly pass `--path ./outputs` when launching training, because the default value in the current training script is machine-specific.
- Training automatically uses CUDA when a compatible GPU is available; otherwise it falls back to CPU.

## Citation

If you use this code or the released datasets in your research, please cite the corresponding GeoSeg-SAM paper. The BibTeX entry will be added after publication.

## Acknowledgements

This work builds upon the Segment Anything Model (SAM) and related open-source research resources. We thank the respective authors and contributors for making their work publicly available.
