# CPE179-E01-OCR_and_Mathematical_Equation_Recognition_System

# Raspberry Pi-Based OCR and Mathematical Equation Recognition System Using TensorFlow Lite

## Overview

This project aims to develop a Raspberry Pi-based Optical Character Recognition (OCR) and Mathematical Equation Recognition System using TensorFlow and TensorFlow Lite. The system will be capable of extracting printed text and mathematical equations from images and converting them into digital text.

Primary focus on text recognition, this project incorporates mathematical equation recognition, enabling the detection of fractions, exponents, symbols, and complex mathematical structures. The entire system is designed to operate locally on a Raspberry Pi without relying on cloud-based services.

---

## Features

* OCR recognition for printed text
* Mathematical equation recognition
* TensorFlow-based machine learning model
* TensorFlow Lite optimized deployment
* Raspberry Pi compatibility
* Local image processing
* Export recognized results
* Lightweight embedded AI implementation

---

## Project Objectives

The project seeks to:

1. Develop an OCR system capable of recognizing printed text.
2. Recognize mathematical equations from images.
3. Deploy a TensorFlow Lite model on Raspberry Pi hardware.
4. Perform image processing and inference locally.
5. Convert recognized content into editable digital format.
6. Demonstrate embedded AI implementation for educational applications.

---

## System Architecture

The system follows a 3-tier architecture:

### Presentation Layer

* Graphical User Interface (GUI)
* Image upload interface
* Result display interface

### Application Layer

* Image preprocessing
* OCR processing
* Mathematical equation recognition
* TensorFlow Lite inference engine

### Data Layer

* Local file storage
* Exported text documents
* Saved recognition results

---

## Workflow

1. User uploads an image.
2. Image preprocessing is performed using OpenCV.
3. OCR extracts standard text.
4. TensorFlow Lite model recognizes mathematical equations.
5. Results are displayed to the user.
6. User can export and save recognized output.

### Sequence Flow

```text
## Technologies Used

### Programming Language

* Python

### Machine Learning

* TensorFlow
* TensorFlow Lite

### Computer Vision

* OpenCV

### OCR

* Tesseract OCR

### Libraries

* NumPy

### Development Environment

* Raspberry Pi OS
* Jupyter Notebook
* Visual Studio Code

---

## Image Preprocessing Techniques

The system utilizes several preprocessing methods to improve recognition accuracy:

* Noise Filtering
* Adaptive Thresholding
* Image Resizing
* Contrast Enhancement
* Character Segmentation

---

## Model Development

The machine learning pipeline includes:

1. Data collection and preparation
2. Image preprocessing
3. Feature extraction
4. OCR model training using TensorFlow
5. Mathematical equation recognition training
6. TensorFlow Lite conversion
7. Model optimization through quantization
8. Raspberry Pi deployment

Attention-based OCR models may be used to improve recognition performance on complex mathematical expressions.

---

## Installation

### Prerequisites

* Raspberry Pi 4 (Recommended)
* Raspberry Pi OS
* Python 3.9+
* TensorFlow Lite Runtime
* OpenCV
* Tesseract OCR

### Clone the Repository

```bash
git clone https://github.com/UyAaron112/CPE179P_TESTPROJECT_GROUP6.git
cd CPE179P_TESTPROJECT_GROUP6
```

### Install Dependencies

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install pytesseract
```

For Raspberry Pi:

```bash
pip install tflite-runtime
```

---

## Usage

### Run the Application

```bash
gui.py
```

### Steps

1. Launch the application.
2. Upload an image containing text or mathematical equations.
3. Wait for processing and recognition.
4. View extracted text and equations.
5. Save or export results.

---

## Evaluation Metrics

The system performance will be evaluated using:

* OCR Accuracy
* Equation Recognition Accuracy
* Precision
* Recall
* F1-Score
* Inference Speed

---

## Team Members

* Gary Botin Jr.
* Eruel Joseph Dizon
* Aaron Dominique Uy

---
