# Fast Few-Iteration Meta-Learning for Few-Shot Classification
This repository contains the code for the paper:
<br>
[Fast Few-Shot Classification by Few-Iteration Meta-Learning
](https://drive.google.com/file/d/1-SjkQZL1Eedg7T6KLQD0go5lzFYzljHG/view?usp=sharing)
<br>
Ardhendu Shekhar Tripathi, Martin Danelljan, Radu Timofte, Luc Van Gool   
ICRA 2021

### Abstract

Autonomous agents interacting with the real world need to learn new concepts efficiently and reliably. This requires learning in a low-data regime, which is a highly challenging problem. We address this task by introducing a fast optimization-based meta-learning method for few-shot classification. It consists of an embedding network, providing a general representation of the image, and a base learner module. The latter learns a linear classifier during the inference through an unrolled optimization procedure. We design an inner learning objective composed of (i) a robust classification loss on the support set and (ii) an entropy loss, allowing transductive learning from unlabeled query samples. By employing an efficient initialization module and a Steepest Descent based optimization algorithm, our base learner predicts a powerful classifier within only a few iterations. Further, our strategy enables important aspects of the base learner objective to be learned during meta-training. To the best of our knowledge, this work is the first to integrate both induction and transduction into the base learner in an optimization-based meta-learning framework. We perform a comprehensive experimental analysis, demonstrating the speed and effectiveness of our approach on four few-shot classification datasets.

## Dependencies
* Python 3.6+
* [PyTorch 1.1.0+](http://pytorch.org)
* [qpth 0.0.11+](https://github.com/locuslab/qpth)
* [tqdm](https://github.com/tqdm/tqdm)

## Usage

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/4rdhendu/FIML.git
    cd FIML
    ```
2. Download and decompress dataset files.

3. For each dataset loader, specify the path to the directory. For example, in FIML/data/mini_imagenet.py, specify:
    ```python
    _MINI_IMAGENET_DATASET_DIR = 'path/to/miniImageNet'
    ```

### Training
1. To train FIML on 5-way miniImageNet benchmark with ResNet backbone:
    ```bash
    python train.py --gpu 0,1,2,3 --save-path "./experiments/miniImageNet_FIML" --train-shot 15 \
    --head FIML --network ResNet_DC --dataset miniImageNet --eps 0.1 --learn-rate 0.1 --val-shot 5

    ```

### Fine-Tuning
1. For fine-tuning the shot specific hyperparameters of the trained model:
```
python train_finetune.py --gpu 0,1,2,3 --load "./experiments/miniImageNet_FIML/best_model.pth" --save-path "./experiments/miniImageNet_FIML" --train-shot 5 \
--head FIML --network ResNet_DC --dataset miniImageNet --eps 0.1 --val-shot 5
```

### Testing
1. To test FIML on 5-way miniImageNet 5-shot benchmark:
```
python test.py --gpu 0,1,2,3 --load ./experiments/miniImageNet_FIML/best_model.pth --episode 1000 \
--way 5 --shot 5 --query 15 --head FIML --network ResNet_DC --dataset miniImageNet
```

All the results in the paper can be reproduced by trying out different options in the run scripts.
## Acknowledgments

This code is based on the implementation of [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet).

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{lee2019meta,
  title={Fast Few-Shot Classification by Few-Iteration Meta-Learning},
  author={Ardhendu Shekhar Tripathi, Martin Danelljan, Radu Timofte and Luc Van Gool},
  booktitle={ICRA},
  year={2021}
}
```
