# NDH-FULL
Code and dataset for EMNLP 2021 paper [“NDH-FULL: Learning and Evaluating Navigational Agents on Full-Length Dialogue”](https://aclanthology.org/2021.emnlp-main.518/) Hyounghun Kim, Jialu Li, and Mohit Bansal.

### Simulator Setup:
Please follow the steps in [this repo](https://github.com/mmurray/cvdn) for setting up the simulator.

## Prerequisites

- Python 3.6
- [PyTorch 1.8](http://pytorch.org/) or Up

## Visual Feature
```
mkdir img_features
```
Please download [this](https://drive.google.com/file/d/1IfI6plfNnXrWVQ4Hmagl-oDSDQnmz_RG/view?usp=sharing) and put it in img_features folder.


## Usage

To train the model:
```
python tasks/NDH_full/train.py --path_type=trusted_path --feedback=sample --eval_type=val --batch_size BATCH_SIZE  --angle_feat_size 128 --feat_type resnet --reinforce
```

## Acknowledgments
Thank you Hao Tan for sharing the CLIP feature.
