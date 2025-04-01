# ResNet18 Weld Defects Classification

This repository contains a PyTorch implementation of a ResNet18 model for classifying weld defects. The model is trained on a custom dataset of weld images to detect various types of aesthetic defects.

## Overview

The project uses a pre-trained ResNet18 model, a lightweight variant of the ResNet architecture, which is well-suited for image classification tasks. The model is fine-tuned on a custom dataset of weld images to classify different types of weld defects.

## Dataset

The dataset consists of images of welds, categorized into different classes based on the type of defect. The dataset is divided into three sets:
- **Training set**: Used to train the model.
- **Validation set**: Used to tune the model and prevent overfitting.
- **Test set**: Used to evaluate the final performance of the model.

The images are resized to 300x30 pixels and normalized using the mean and standard deviation of the ImageNet dataset.

## Model Architecture

The model used in this project is based on the ResNet18 architecture, which consists of 18 layers. ResNet18 leverages residual blocks to address the vanishing gradient problem, enabling efficient training of deeper networks.

### Modifications to ResNet50

The final fully connected layer of the ResNet18 model is modified to match the number of classes in the custom dataset. This adaptation allows the model to output predictions for the specific defect classes in the dataset.

## Training

The model is trained using the following configuration:
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with a learning rate of 0.001
- **Number of Epochs**: 10

During training, the model's performance is monitored on both the training and validation sets. The training process includes:
- Forward pass to compute the output and loss.
- Backward pass to compute gradients.
- Optimization step to update the model's weights.

## Evaluation

After training, the model is evaluated on the test set to measure its performance. The evaluation metrics include:
- **Test Loss**: The average loss over the test set.
- **Test Accuracy**: The percentage of correctly classified images in the test set.

## Results

The model achieves the following performance during training and validation:
- **Test Loss**: 91.50% (final epoch)
- **Test Accuracy**: 90.55% (final epoch)

## Usage

To use this model, follow these steps:

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/resnet18_weld_defects.git
cd resnet18_weld_defects
```
   
2. **Clone the repository**:
```bash
pip install torch torchvision tqdm matplotlib
```
   
3. **Clone the repository**:
```bash
python resnet18_weld_defects.ipynb
```
