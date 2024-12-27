# OpenCV & AI
## About
This repository contains a collection of projects for exploring and implementing medical image processing techniques and machine learning models. The projects are mainly based on the OpenCV and PyTorch libraries. 

## Installation
In order to run the program, you will need to install Python3.12 and pip. Then install the required dependencies by typing:
```bash
pip install -r requirements.txt
```

## Projects
### 1. [Lungs segmentation](lab01_lungs/Project1.ipynb)
The aim of the project is to segment the lungs from the 3D chest X-ray images. The version with the dataset is available [here](https://github.com/lursz/OpenCVAndAI/tree/with-datasets). The dataset contains images in the DICOM format. The project is divided into two parts: `body mask segmentation` and `lungs segmentation`.  

`Body mask segmentation` includes cutting out the body from the image using binarization (with custom threshold) and reconstruction, and then comparing the results with the ground truth. 

`Lungs segmentation` on the other hand includes cutting the lungs from the image and comparing the results with the ground truth. 

### 2. Turning Lungs segmentation into [3D Slicer](https://www.slicer.org/) plugin

### 3. [Autoencoder](lab03_autoencoder/Autoencoder.ipynb)
An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data. 

![denoise autoencoder](https://upload.wikimedia.org/wikipedia/commons/1/18/Denoising-autoencoder.png)

In this case the autoencoder is used to convert scans into a lower-dimensional representation (in other words, compressing the data) and then reconstructing the original image from the compressed representation. However, reconstruction can be tweaked to generate images that are not the same as the original ones. In this case, graphics of ellipses with openings are generated, and then the encoder and decoder are trained to encode and reconstruct the original image, but this time without the openings.


### 4. [Abdomen detection](lab04_abdomen/abdomen.ipynb)
In this project, a NN was created to detect the abdomen from the 3D body X-ray images. The version with the dataset is available [here](https://github.com/lursz/OpenCVAndAI/tree/with-datasets). Neural network loops through the images and decides whether the current slice may be considered as an abdomen or not. The result is a 1-dimensional array of probabilities. Then the array is passed into a function that rejects the outliers and returns the final result - window of the abdomen.