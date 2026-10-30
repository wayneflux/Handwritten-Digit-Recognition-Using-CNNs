# Handwritten Digit Recognition Using CNNs

A deep learning project that uses a Convolutional Neural Network (CNN) built with PyTorch to recognize handwritten digits from 0 to 9. The project uses the MNIST dataset to train a neural network that learns visual patterns in handwritten digits and predicts the digit represented by an image.

Beyond training and evaluating the model, the project includes functionality for predicting digits from external image files and an interactive drawing interface built with Pygame. Users can draw a digit and let the trained model predict what they have drawn.

## Table of Contents

* [Project Overview](#project-overview)
* [Project Objectives](#project-objectives)
* [Technologies Used](#technologies-used)
* [Dataset](#dataset)
* [CNN Architecture](#cnn-architecture)
* [Model Training](#model-training)
* [Image Prediction](#image-prediction)
* [Interactive Digit Recognition](#interactive-digit-recognition)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Project Structure](#project-structure)
* [Challenges and Limitations](#challenges-and-limitations)
* [Future Improvements](#future-improvements)
* [Key Learnings](#key-learnings)
* [Author](#author)

## Project Overview

Handwritten digit recognition is a computer vision task that involves identifying numerical digits from images.

Although handwritten digits represent only ten possible classes, recognizing them automatically is challenging because handwriting varies considerably between individuals. The same digit can have different shapes, sizes, orientations, and stroke patterns.

This project explores how deep learning can address this problem using a Convolutional Neural Network.

The model is trained using the MNIST dataset, which contains grayscale images of handwritten digits. During training, the CNN learns useful visual features that allow it to distinguish between the ten digit classes.

After training, the model can predict digits from image files. The project also incorporates a Pygame drawing interface that allows users to draw digits and obtain predictions interactively.

The project demonstrates an end-to-end deep learning workflow, from dataset preparation and model construction to training, saving model parameters, and making predictions on new images.

## Project Objectives

The main objectives of this project are to:

* Understand the practical implementation of Convolutional Neural Networks.
* Apply deep learning to a multiclass image classification problem.
* Load and prepare the MNIST dataset using PyTorch and Torchvision.
* Build a CNN with convolutional layers, pooling, dropout, and fully connected layers.
* Train the model using an appropriate optimizer and loss function.
* Save trained model parameters for subsequent inference.
* Develop a function for predicting digits from external image files.
* Create an interactive drawing interface using Pygame.
* Explore the challenges involved in applying a trained model to images that differ from its training data.

## Technologies Used

| Technology       | Purpose                                          |
| ---------------- | ------------------------------------------------ |
| Python           | Primary programming language                     |
| PyTorch          | Building and training the neural network         |
| Torchvision      | Loading MNIST and preparing image tensors        |
| NumPy            | Numerical operations                             |
| Matplotlib       | Image visualization                              |
| Pillow (PIL)     | Loading and processing images                    |
| Pygame           | Creating the interactive digit-drawing interface |
| KaggleHub        | Downloading a dataset resource                   |
| Google Colab     | Cloud-based development environment              |
| Jupyter Notebook | Interactive development and experimentation      |

## Dataset

The project uses the **MNIST handwritten digit dataset**, a widely used dataset for training and evaluating image classification models.

MNIST contains grayscale images representing handwritten digits from 0 through 9.

Each image has dimensions of 28 × 28 pixels and belongs to one of ten classes.

### Dataset Characteristics

* **Input:** Grayscale image.
* **Image dimensions:** 28 × 28 pixels.
* **Number of classes:** 10.
* **Class labels:** Integers from 0 to 9.
* **Task:** Multiclass image classification.

### Dataset Loading

The notebook loads MNIST using `torchvision.datasets.MNIST`.

The dataset-loading configuration enables automatic downloading when the dataset is not already available locally.

The notebook also includes a KaggleHub dataset download operation.

### Data Preparation

The images are converted into tensors using Torchvision transformations.

The `ToTensor()` transformation converts image data into a tensor representation and scales the pixel values to the range [0, 1].

The training and test datasets are loaded separately, allowing the model to be trained on one dataset and potentially evaluated on another.

## CNN Architecture

The project implements a custom Convolutional Neural Network named `DigitClassifier` using PyTorch.

CNNs are particularly effective for image classification because convolutional layers can learn spatial features directly from image data.

The network uses multiple layers to transform the input image into a representation that can be used to predict the digit class.

### Architecture Components

**1. First convolutional layer**

The first convolutional layer contains 32 output channels and is followed by a ReLU activation function.

It learns basic spatial features from the input image.

**2. Second convolutional layer**

The second convolutional layer contains 64 output channels and is also followed by ReLU.

It extracts additional features from the representations produced by the preceding layer.

**3. Max-pooling layer**

The pooling layer reduces the spatial dimensions of the feature maps.

This helps reduce the number of values passed to subsequent layers while retaining useful spatial information.

**4. Dropout**

Dropout is used to reduce over-reliance on individual neurons during training.

It randomly deactivates a proportion of activations during training, helping regularize the network.

**5. Flattening**

The extracted feature maps are flattened into a one-dimensional representation suitable for fully connected layers.

**6. Fully connected layer**

A dense layer containing 128 units processes the extracted features and learns relationships useful for distinguishing between the digit classes.

A ReLU activation function introduces nonlinearity.

**7. Output layer**

The final layer contains ten output units, corresponding to digits 0 through 9.

The model produces a score for each class. The predicted digit is selected by identifying the class with the highest score.

### Architecture Summary

| Component                    | Configuration      |
| ---------------------------- | ------------------ |
| First convolutional layer    | 32 output channels |
| Second convolutional layer   | 64 output channels |
| Activation function          | ReLU               |
| Pooling                      | Max pooling        |
| Regularization               | Dropout            |
| Fully connected hidden layer | 128 units          |
| Output layer                 | 10 units           |
| Number of classes            | 10                 |

## Model Training

The model is trained using PyTorch.

During training, batches of images are passed through the network to produce predictions. The loss function measures the difference between the predicted scores and the actual digit labels.

The optimizer then updates the model's parameters to reduce the classification loss.

### Training Configuration

| Parameter               | Configuration             |
| ----------------------- | ------------------------- |
| Deep learning framework | PyTorch                   |
| Optimizer               | Adam                      |
| Loss function           | Cross-entropy loss        |
| Training batch size     | 64                        |
| Number of epochs        | 30                        |
| Classification type     | Multiclass classification |

### Optimizer

The Adam optimizer is used to update the network's parameters during training.

Adam adapts parameter updates using estimates of gradients and their historical behavior.

It is widely used in deep learning because it provides an adaptive optimization method.

### Loss Function

The model uses `CrossEntropyLoss`.

Cross-entropy loss is suitable for multiclass classification tasks in which a model produces scores for several mutually exclusive classes.

In this project, the ten output units correspond to the ten possible digit classes.

The model learns to assign higher scores to the correct digit class.

### Training Process

The training process involves:

1. Loading batches of training images and their labels.
2. Passing images through the CNN.
3. Calculating the classification loss.
4. Computing gradients through backpropagation.
5. Updating the model parameters using Adam.
6. Repeating the process across training batches and epochs.
7. Saving the trained model parameters.

The notebook runs training for 30 epochs.

**Note:** The notebook does not currently report a measured test accuracy or a confusion matrix. These metrics should be calculated before making claims about the model's final performance.

## Image Prediction

The project includes a `predict_digit(image_path, model)` function for classifying digits from image files.

This allows the trained model to be used on images beyond those directly supplied by the MNIST dataset loader.

### Prediction Workflow

The prediction function performs the following operations:

1. Opens the supplied image.
2. Converts the image to grayscale.
3. Resizes the image to 28 × 28 pixels.
4. Converts the image into a tensor.
5. Applies normalization.
6. Passes the processed image through the trained CNN.
7. Identifies the class with the highest output score.
8. Returns the predicted digit.

The predicted class is an integer between 0 and 9.

### Example Use Case

A user can supply an image containing a handwritten digit, and the model attempts to identify the numerical value represented by that image.

For example, an image containing a handwritten representation of the digit 5 should ideally produce a prediction of `5`.

The result depends on the model's learned features and how closely the supplied image resembles the images used during training.

### Important Preprocessing Consideration

The notebook's training pipeline uses `ToTensor()`, while the custom-image prediction function additionally applies normalization using a mean of 0.5 and a standard deviation of 0.5.

This creates a difference between training and inference preprocessing.

For more reliable predictions, the preprocessing pipelines should be made consistent. Custom images may also require adjustments to foreground polarity, centering, and background appearance.

## Interactive Digit Recognition

The project includes an interactive drawing interface built with Pygame.

The interface allows a user to draw a digit using the mouse and submit the drawing to the trained CNN for prediction.

This provides a more interactive way to experiment with the model than supplying image files manually.

### How It Works

1. The Pygame application opens a drawing window.
2. The user draws a digit on the canvas.
3. The drawing can be cleared when necessary.
4. The user requests a prediction.
5. The application saves the drawing as `drawn_digit.png`.
6. The image is processed and passed to the trained model.
7. The predicted digit is printed and the processed image is displayed.

### Keyboard Controls

| Key | Function                 |
| --- | ------------------------ |
| `C` | Clear the drawing canvas |
| `P` | Predict the drawn digit  |
| `Q` | Quit the application     |

### Practical Considerations

The MNIST dataset has specific image characteristics, including image dimensions and typical foreground and background patterns.

Digits drawn on a custom canvas may differ from the training images in stroke thickness, position, scale, and contrast.

Consequently, a model that performs well on MNIST images may make mistakes when predicting user-drawn digits.

The Pygame interface also requires a graphical environment. Running it in a local desktop Python environment is generally more suitable than running it directly in a standard Google Colab notebook session.

## Installation

To run the project locally, install Python and the required libraries.

### 1. Clone the Repository

Replace `YOUR-GITHUB-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/handwritten-digit-recognition-cnn.git
```

Navigate into the project directory:

```bash
cd handwritten-digit-recognition-cnn
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision kagglehub Pillow matplotlib numpy pygame jupyter
```

The appropriate PyTorch installation may vary depending on the operating system and whether CPU or CUDA acceleration is required.

## Running the Project

The notebook was developed using Google Colab.

### Running in Google Colab

1. Open the notebook in Google Colab.
2. Execute the cells in order.
3. Allow the dataset-loading operations to complete.
4. Run the model definition and training cells.
5. Save the trained model parameters.
6. Execute the image-prediction function with a valid image path.

### Running Locally

1. Clone the GitHub repository.
2. Install the required dependencies.
3. Open the notebook using Jupyter Notebook or JupyterLab.
4. Execute the cells in order.
5. Ensure the dataset and image paths are valid.
6. Run the training or prediction sections as required.

The notebook saves the trained model parameters in:

`mnist_digit_classifier.pth`

The prediction workflow reconstructs the model architecture before loading the saved parameters.

The architecture used to load the parameters must match the architecture used when they were saved.

The image-prediction example also uses an image path such as `testSet/img_5.jpg`. Replace this with a path to an image that actually exists in your environment.

## Project Structure

A possible repository structure is shown below:

```text
handwritten-digit-recognition-cnn/
│
├── README.md
├── main.ipynb
├── mnist_digit_classifier.pth
├── drawn_digit.png
└── .gitignore
```

The actual repository may differ from this example.

* `README.md`: Project documentation.
* `main.ipynb`: Notebook containing the implementation.
* `mnist_digit_classifier.pth`: Saved model parameters generated during training.
* `drawn_digit.png`: An image generated by the drawing interface.
* `.gitignore`: Specifies files that Git should ignore.

The dataset does not need to be committed to the repository.

## Challenges and Limitations

Several considerations affect the reliability and reproducibility of this project.

### 1. Training and Inference Preprocessing

The training and custom-image prediction pipelines use different normalization procedures.

This should be corrected to ensure that the model receives inputs with consistent numerical properties.

### 2. Custom Image Variability

External images may differ from MNIST images in size, contrast, orientation, stroke thickness, and digit placement.

These differences can affect prediction performance.

### 3. Evaluation Metrics

The notebook does not currently report test accuracy, precision, recall, F1-score, or a confusion matrix.

Adding these measurements would provide a more rigorous assessment of the trained model.

### 4. Graphical Interface Compatibility

The Pygame interface requires a graphical display environment and may not run directly inside a standard hosted notebook environment.

### 5. Reproducibility

Training results may vary depending on random initialization, dataset handling, library versions, and hardware.

Recording the environment and training configuration would make future experiments easier to reproduce.

## Future Improvements

The project can be extended in several ways.

* **Model evaluation:** Calculate test accuracy and test loss using the MNIST test dataset.
* **Performance visualization:** Plot training loss and accuracy across epochs.
* **Confusion matrix:** Identify which digit classes are most frequently misclassified.
* **Prediction confidence:** Display the model's class scores or probabilities alongside the predicted digit.
* **Image preprocessing:** Improve digit centering, resizing, contrast, and foreground-background handling.
* **Preprocessing consistency:** Apply the same normalization convention during training and inference.
* **Model comparison:** Compare the custom CNN with alternative architectures.
* **User interface:** Improve the drawing canvas and display predictions directly within the application.
* **Deployment:** Package the classifier as a desktop application or a web-based digit recognition service.
* **Model persistence:** Document the saved model format and provide a reusable loading function.

These improvements would help turn the notebook into a more robust and thoroughly evaluated digit recognition system.

## Key Learnings

This project provides practical experience with several important deep learning concepts.

### Convolutional Neural Networks

Understanding how convolutional layers extract spatial features from image data.

### Multiclass Classification

Using a neural network to distinguish between ten possible output classes.

### Backpropagation and Optimization

Understanding how gradients and optimizers update model parameters during training.

### Data Preprocessing

Learning why image dimensions, grayscale conversion, tensor representation, and normalization matter.

### Model Persistence

Saving model parameters and loading them for subsequent inference.

### Computer Vision Inference

Applying a trained CNN to image files outside the training pipeline.

### Interactive Machine Learning

Connecting a trained model to a graphical interface that allows users to draw digits and obtain predictions.

## Author

Developed as a personal deep learning project to strengthen practical understanding of convolutional neural networks, image classification, and model inference using Python and PyTorch.

## Acknowledgements

* The creators and maintainers of the MNIST handwritten digit dataset.
* The PyTorch and Torchvision communities.
* The Pygame project.
* Google Colab for providing a cloud-based notebook environment.

---

**Disclaimer:** This project is intended for educational purposes. Predictions are not guaranteed to be correct, particularly for handwritten images that differ substantially from the training data.
