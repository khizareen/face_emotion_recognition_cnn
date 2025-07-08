# face_emotion_recognition_cnn
A deep learning-based face emotion recognition system using CNNs on the FER-2013 dataset. The model classifies images into 7 emotions with data augmentation, class balancing, evaluation via classification metrics, and real-time image predictions. Built using TensorFlow/Keras.

# Face Emotion Recognition using CNN

This project implements a deep learning model using Convolutional Neural Networks (CNNs) to recognize emotions from facial expressions. The model is trained on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset and is capable of classifying images into 7 emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

## Dataset

The dataset is divided into `train/` and `test/` folders, each containing subdirectories for the seven emotion classes. Images are 48x48 grayscale facial expressions.
Make sure your data is extracted from `fer_data_set.zip` into a `fer_data` folder.

## Model Architecture

The CNN model includes:
- Three convolutional blocks (with Conv2D, BatchNormalization, MaxPooling, Dropout)
- Global Average Pooling
- Dense layer with dropout
- Softmax output for 7 classes

Optimizer: `Adam`  
Loss: `Categorical Crossentropy`  
Metrics: `Accuracy`

## Features

- Data Augmentation using `ImageDataGenerator`
- Class imbalance handling using `class_weight`
- EarlyStopping and ReduceLROnPlateau callbacks
- Evaluation: Accuracy, Loss, Classification Report, and Confusion Matrix
- Real-time image prediction from uploaded images
- Model saving (`fer_cnn_model.h5`)

## Training and Evaluation

- Trained for 50 epochs
- Validation split: 20% of training data
- Accuracy and loss visualized for both training and validation sets
- Final test accuracy and detailed performance metrics provided

## Results

- Achieved high test accuracy (varies by training)
- Classification report and confusion matrix used for performance insight
- Can generalize well to unseen facial images

## Emotion Prediction on Custom Image

After training:

1. Upload the trained model (`fer_cnn_model.h5`)
2. Upload a facial image
3. Model will display the image and print the predicted emotion label

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn
- OpenCV
- PIL (Pillow)

## How to Run

1. **Extract the Dataset**
import zipfile
zip_path = 'fer_data_set.zip'
extract_to = 'fer_data'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

2. **Train the Model**
Run the training script to preprocess data, build the model, and begin training.

3. **Evaluate & Save**
Model is saved automatically as fer_cnn_model.h5.

4. **Predict**
Upload your own image and get emotion prediction in real-time.

## Installation
Install the required dependencies:
pip install tensorflow keras opencv-python pillow matplotlib seaborn scikit-learn

## Folder Structure
fer_data/
  ├── train/
  │    ├── Angry/
  │    ├── Disgust/
  │    └── ...
  └── test/
       ├── Angry/
       ├── Disgust/
       └── ...

## Author
Khizareen Taj
Feel free to fork or contribute!
