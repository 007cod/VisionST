<div align="center">
  <h2><b> VisionST: Coordinating Cross-modal Traffic Prediction with Interactive Geo-image Encoding
 </b></h2>
</div>

## Introduction
VisionST coordinates cross-modal traffic prediction with interactive geo-image encoding.
Notably, we show that spatiotemporal graphs can be patched on the spatial dimension, effectively reducing complexity in attention.

<p align="center">
<img src="./image/frame.png" height = "300" alt="" align=center />
</p>

- VisionST is composed of three main components: (1) Multi-modal Embedding, which consists of spatiotemporal embedding and image embedding, aims to transform the traffic data into a high-dimensional representation, thereby facilitating more effective learning of complex patterns. (2) Vision-Augmented Layer, which extracts node-level visual tokens from geo-images and integrates them into spatiotemporal representations, enriching the feature space with localized environmental context. (3) Pattern Interaction Layer, which generates relation patterns that encompass visual, spatial, and temporal aspects, constrains nodes to interact with them for contextual information interaction.

## Requirements
- torch==2.6.0
- timm==1.0.15
- tqdm==4.67.1
- pandas==2.2.3
- numpy==2.2.3
- tensorboard==2.19.0

## Folder Structure

```tex
└── code-and-data
    ├── config                 # Including detail configurations
    ├── cpt                    # Storing pre-trained weight files (manually create the folder and download files)
    ├── data                   # Including traffic data (download), adj files (generated), geo-image data, and the meta data
    ├── lib
    │   |──  utils.py          # Codes of preprocessing datasets and calculating metrics
    ├── log                    # Storing log files
    ├── model
    │   |──  visionst.py         # The core source code of our visionst
    │   |──  layer.py           
    │   |──  vis.py             # The core source code of our vision model
    ├── main.py                # This is the main file for training and testing
    └── README.md              # This document
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/1BDH1C66BCKBe7ge8G-rBaj1j3p0iR0TC?usp=sharing), then place the downloaded contents under the correspond dataset folder such as `./data/SD`.

## Quick start
1. Download datasets and place them under `./data`
2. Install the required Python packages by running:
  ```
  pip install -r requirements.txt
  ```
3. We provide the detail configurations under the folder `./config`. For example, you can train the SD dataset by:

```
python main.py --config ./config/SD.conf
```
