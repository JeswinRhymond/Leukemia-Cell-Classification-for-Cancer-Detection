Leukemia-Cell-Classification-for-Cancer-Detection

Overview
The Leukemic Cell Classification Project aims to develop a Convolutional Neural Network (CNN) for classifying leukemia cells to aid in cancer detection. It involves three key phases: dataset preparation, model training, and a prediction system. The dataset, sourced from a public medical image repository, is restructured into leukemic (ALL) and non-leukemic (HEM) categories, consolidating images for efficient training. A custom CNN model is designed with convolutional, pooling, dense, and dropout layers, compiled using the Adam optimizer and binary crossentropy loss. Data augmentation is applied to improve generalization, and training incorporates callbacks like early stopping and learning rate reduction. The model's performance is evaluated using accuracy/loss graphs, a classification report, and a confusion matrix. The model is saved for deployment after it has been trained. The prediction system includes a user-friendly GUI for image upload, preprocessing, and classification, displaying results with probabilities and enabling image visualization and deletion. This project provides an automated, robust, and intuitive solution for leukemia cell classification, showcasing its potential for effective cancer detection.

Dataset
Our dataset comprises a comprehensive collection of 15,135 segmented cell images obtained from 118 patients. Each image has undergone meticulous annotation by expert oncologists, providing valuable ground truth labels for training and evaluation. The dataset accurately reflects real-world scenarios, encompassing staining noise and illumination errors often encountered in microscopic imaging. Rigorous efforts have been made to rectify these errors during the image acquisition process to ensure dataset reliability and quality.

Dataset Features:
Image Format: .bmp Color Encoding: 8-bit RGB Data Splits: Train, Test, Validation Training Folds: fold_0, fold_1, fold_2 Folders: all (images with leukemia cells), hem (images without cells or leukemia)

Methodology
Feature Extraction and Preprocessing Prior to classification, the dataset undergoes preprocessing to enhance image quality and extract crucial features. Attributes like cell shape, texture, and spatial connections are extracted to capture the properties of leukemic cells. These features provide the foundation for the subsequent machine learning algorithms.

Algorithms Utilized
Random Forest
K-Nearest Neighbors (KNN):
Support Vector Machines (SVM)
Naive Bayes
Convolutional Neural Networks (CNN):
Evaluation
The performance of each algorithm is assessed using appropriate metrics such as accuracy, precision, recall, and F1 score. Through rigorous testing and comparison, the Convolutional Neural Network (CNN) algorithm has demonstrated itself as the most suitable choice for this project, given its proficiency in medical picture analysis and classification tasks.

Objective
The primary objective of this project is to create an automated and reliable system for leukemic cell classification. Accurate identification and classification of leukemic cells can aid in early detection and timely treatment of leukemia, potentially improving patient outcomes, particularly for children diagnosed with Acute Lymphoblastic Leukemia (ALL).

Contribution
By enabling automated and reliable leukemia cell classification, this project aspires to make significant contributions to the field of medical image analysis. The outcomes have the potential to revolutionize clinical decision-making, enable personalized treatment approaches, and ultimately enhance the prognosis for leukemia patients.
