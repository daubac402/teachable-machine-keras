# Image Classification With Pre-Trained Keras Model

This Python script is designed for image classification using a pre-trained deep learning model. It reads images from an input folder, predicts their class labels, and organizes them into class-specific folders in an output directory.

## Features

- Loads a pre-trained Keras model and class labels.
- Processes images (resizing and normalization).
- Predicts the class of each image.
- Moves images to class-specific folders based on predictions.
- Supports a confidence threshold to handle uncertain predictions.

## Prerequisites

- Python 3.x
- Required Python packages: Keras, Pillow (PIL), NumPy or Miniconda to manage them at once

## Usage

### First-time setup

#### Create the folder containing input images

```bash
mkdir -p images/input
```

Then put some images into that folder.

#### Customize the following constants in the main.py script

- `MODEL_PATH`: Path to the pre-trained Keras model file.
- `LABEL_PATH`: Path to the file containing class labels.
- `INPUT_IMAGE_FOLDER`: Path to the folder containing input images.
- `OUTPUT_IMAGE_FOLDER`: Path to the folder where organized images will be saved.
- `NOT_SURE_CLASS`: Default class name for images with low confidence.
- `NOT_SURE_THRESHOLD`: Confidence threshold for classifying as "Not Sure."

#### Install the required Python packages

```bash
pip install keras tensorflow Pillow
```

Or, pretty much convenient way with Miniconda:

```bash
conda env create -f environment.yml
```

### Start the script

```bash
# If you use Miniconda, activate the environment first.
conda activate keras-tf-pillow

python main.py
```
