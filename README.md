# Traffic Sign Detection Web Application

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This project is a **Traffic Sign Detection** web application built using Machine Learning techniques. The app automates the recognition of traffic signs from images using a Convolutional Neural Network (CNN) model, improving accuracy, efficiency, and safety compared to manual recognition.

Traffic signs are essential for safe driving, but manual recognition can be error-prone and slow. This application aims to solve these problems by providing real-time automated traffic sign identification and an educational platform to understand the underlying machine learning concepts.

## Features

- **Automated Traffic Sign Recognition** using a trained CNN model.
- **Real-time predictions** upon image upload.
- **Educational content** explaining the machine learning principles like CNNs and image preprocessing.
- **Model insights** explaining how the CNN model makes predictions.
- **Handles varying image inputs** including images with alpha transparency.
- **Accessible and user-friendly web interface.**

## Project Highlights

- Utilizes the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset from Kaggle for training and testing.
- Implements a deep CNN architecture with convolutional, pooling, dense, and dropout layers for robust learning.
- Optimizes training with the Adam optimizer and dropout regularization to reduce overfitting.
- Provides detailed preprocessing including resizing and RGB channel management.
- Emphasizes ethical AI development including bias mitigation, transparency, and accessibility.

## How It Works

1. **User uploads an image** of a traffic sign through the web app interface.
2. The image is **preprocessed** (resized to 30x30 pixels, RGB channels normalized).
3. The **trained CNN model** predicts the traffic sign label.
4. The prediction is displayed immediately on the web page.
5. Users can also explore educational content about the model and machine learning concepts.

## Technologies Used

- Python
- TensorFlow / Keras (for CNN model training)
- Flask (for web application backend)
- HTML/CSS/JavaScript (frontend)
- GTSRB Dataset (from Kaggle)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-detection.git
   cd traffic-sign-detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000` to use the app.

## Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which includes thousands of traffic sign images categorized into multiple classes.

- Dataset link: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## Ethical Considerations

- Mitigating model bias to ensure fair predictions.
- Providing transparency and explainability of the model.
- Ensuring accessibility for diverse user groups.
- Promoting accountability for prediction accuracy.
- Continuously improving the system with ethical AI best practices.

## Future Work

- Expand support for real-time video stream detection.
- Integrate with vehicle assistance systems.
- Enhance model robustness against adverse lighting and weather conditions.
- Add multilingual educational content.

## Acknowledgements & References

- Dataset: GTSRB - German Traffic Sign Recognition Benchmark (Kaggle)
- CNN and machine learning concepts inspired by tutorials from DataFlair, TensorFlow, and LearnOpenCV.
- Flask web framework documentation.

## Contact

Ankushkumar Rajubhai Lakhani  
Email: ankushkumarrajubhai.lakhani@ontariotechu.net

Thank you for checking out this project! Feel free to contribute or raise issues.
