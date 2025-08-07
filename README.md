# M2ANet
<<<<<<< HEAD

## Abstract
Convolutional neural networks (CNNs)-based medical image segmentation technologies have been widely used in medical image segmentation because of their strong representation and generalization abilities. However, due to the inability to effectively capture global information from images, CNNs can easily lead to loss of contours and textures in segmentation results. Notice that the transformer model can effectively capture the properties of long-range dependencies in the image, and furthermore, combining the CNN and the transformer can effectively extract local details and global contextual features of the image. Motivated by this, we propose a multi-branch and multi-scale attention network (M2ANet) for medical image segmentation, whose architecture consists of three components. Specifically, in the first component, we construct an adaptive multi-branch patch module for parallel extraction of image features to reduce information loss caused by downsampling. In the second component, we apply residual block to the well-known convolutional block attention module to enhance the network’s ability to recognize important features of images and alleviate the phenomenon of gradient vanishing. In the third component, we design a multi-scale feature fusion module, in which we adopt adaptive average pooling and position encoding to enhance contextual features, and then multi-head attention is introduced to further enrich feature representation. Finally, we validate the effectiveness and feasibility of the proposed M2ANet method through comparative experiments on four benchmark medical image segmentation datasets, particularly in the context of preserving contours and textures. The source code of M2ANet will be released at https://github.com/AHUT-MILAGroup/M2ANet

## Experiments

In order to run the experiments, please follow these steps

### Prepare the Dataset

Prepare the train, validation and test split of your dataset.   

Then store all the images and corresponding masks as .png images and save with the same name, in the img and labelcol directories, respectively.

The directory structure should be as follows:

```angular2html
├── datasets
    ├── GlaS_exp1
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── BUSI_exp1
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```

### Train the Model

First, modify the model, dataset and training hyperparameters in `Config.py`

Then simply run the training code.

```
python3 train_model.py
```

**Note:** In order to train Swin-Unet or SMESwin-UNet please collect the pretrained checkpoint from https://github.com/HuCaoFighting/Swin-Unet and put that inside `Experiments/pretrained_ckpt`


### Evaluate the Model

Please make sure the right model and dataset is selected in `Config.py`

Then simply run the evaluation code.

```
python3 test_model.py
```

## Datasets
- BUSI : [https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- ISIC2018 : [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)
- GlaS : [https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation)
- PolS : [https://www.kaggle.com/datasets/debeshjha1/medico-automatic-polyp-segmentation-challenge](https://www.kaggle.com/datasets/debeshjha1/medico-automatic-polyp-segmentation-challenge)


## Cite
```python
@article{10.1088/1674-1056/adce96,
	author={Xue, Wei and Chen, Chuanghui and Qi, Xuan and Qin, Jian and Tang, Zhen and He, Yongsheng},
	title={M2ANet: Multi-branch and multi-scale attention network for medical image segmentation},
	journal={Chinese Physics B},
	url={http://iopscience.iop.org/article/10.1088/1674-1056/adce96},
	year={2025}
}
```
=======
M2MNet
>>>>>>> origin/main
