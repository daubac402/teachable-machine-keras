# An image classification script that uses a pre-trained deep learning model (Keras model trained with Teachable Machine)
# to classify images located in a specified input folder and organizes them
# into class-specific folders in an output folder based on the predicted class labels.

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Define model constants
MODEL_PATH = "model/keras_Model.h5"
LABEL_PATH = "model/labels.txt"
INPUT_IMAGE_FOLDER = "images/input"
OUTPUT_IMAGE_FOLDER = "images/output"
NOT_SURE_CLASS = "NotSure"
NOT_SURE_THRESHOLD = 0.8


# Load the model
def load_keras_model(model_path):
    return load_model(model_path, compile=False)


# Load the labels
def load_class_names(label_path):
    return open(label_path, "r").readlines()


# Preprocess an image
def preprocess_image(image_path):
    size = (224, 224)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array


# Predict the class of an image
def predict_class(model, data, class_names, not_sure_threshold, not_sure_class):
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    class_name = (
        not_sure_class
        if confidence_score < not_sure_threshold
        else class_names[index][2:-1]
    )
    return class_name, confidence_score


# Organize and move images
def organize_images(input_folder, output_folder, class_name, filename):
    output_folder = os.path.join(output_folder, class_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.rename(
        os.path.join(input_folder, filename), os.path.join(output_folder, filename)
    )


# Main function
def main():
    model = load_keras_model(MODEL_PATH)
    class_names = load_class_names(LABEL_PATH)

    for filename in os.listdir(INPUT_IMAGE_FOLDER):
        filename_lower = filename.lower()

        if not (
            filename_lower.endswith(".jpg")
            or filename_lower.endswith(".jpeg")
            or filename_lower.endswith(".png")
        ):
            print(f"Skipping: {filename}")
            continue

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image_path = os.path.join(INPUT_IMAGE_FOLDER, filename)
        normalized_image_array = preprocess_image(image_path)

        data[0] = normalized_image_array

        class_name, confidence_score = predict_class(
            model, data, class_names, NOT_SURE_THRESHOLD, NOT_SURE_CLASS
        )

        print(f"Image: {filename} -> class: {class_name}, score: {confidence_score}")

        organize_images(INPUT_IMAGE_FOLDER, OUTPUT_IMAGE_FOLDER, class_name, filename)


if __name__ == "__main__":
    main()
