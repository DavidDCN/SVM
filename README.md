⚙️ How It Works: The ML Pipeline
1. Data Cleaning & Exploration
The script first scans the directories, extracts metadata (dimensions, file size), and filters out unreadable images or anomalies smaller than 20x20 pixels to ensure high-quality training data.

2. Image Preprocessing
Before feature extraction, each image undergoes a strict standardization process:

Grayscale Conversion: Removes color noise, focusing the model on structural shapes.

Resizing: Standardized to 128x128 pixels.

Normalization: Pixel values are scaled to a [0, 1] range.

Histogram Equalization: Enhances contrast, ensuring the model performs well regardless of the original lighting conditions.

3. Feature Extraction (HOG)
Instead of feeding raw pixels into the model, we extract structural features using HOG (Histogram of Oriented Gradients). HOG excels at capturing edge directions and local shapes (like the rigid edge of a mask or the curve of a chin), ignoring background noise.

4. Dimensionality Reduction (PCA)
HOG generates a massive feature vector. We apply PCA to compress these features down to the top 300 principal components, retaining the most critical variance while drastically reducing training time and preventing overfitting.

5. Classification (SVM)
An SVM with a Radial Basis Function (RBF) kernel is trained on the PCA-reduced features. Because the dataset becomes imbalanced when grouping without_mask and mask_weared_incorrect together, the SVM uses class_weight='balanced' to prevent bias toward the majority class.

🚀 Setup & Installation
Dependencies
Ensure you have Python installed, then install the required libraries:

Bash
pip install numpy pandas opencv-python scikit-learn scikit-image matplotlib seaborn tqdm
Configuration
Before running the script, update the DATASET_FOLDERS dictionary in the CONFIG section of the code to point to your local dataset directories.

Note: The default paths are currently set to the author's local environment:

Python
DATASET_FOLDERS = {
    'with_mask': ,
    # ... update these to your local paths ...
}
📈 Outputs & Artifacts
Running the script will automatically generate and save the following artifacts in your working directory:

Visualizations
confusion_matrix.png: A heatmap showing the True Positives, True Negatives, False Positives, and False Negatives on the 20% test split.

sample_predictions.png: A 2x5 grid of random test images with their predicted labels overlaid (Green = Correct, Red = Incorrect).

Exported Models
facemask_svm_model.pkl: A comprehensive dictionary containing the trained SVM model, PCA transformer, StandardScaler, and evaluation metrics.

preprocessed_data.pkl: The transformed and split datasets (X_train, X_test, etc.) saved for fast experiment iteration without needing to re-extract HOG features.
""")
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 3728-3729: truncated \UXXXXXXXX escape

Plaintext
Dataset/
├── with_mask/                 # Images of properly worn masks
├── without_mask/              # Images of bare faces
└── mask_weared_incorrect/     # Images of improperly worn masks (e.g., nose exposed)
Label Mapping (Strict Compliance)
The system is designed for strict safety compliance. Therefore, incorrectly worn masks are penalized and classified identically to having no mask at all:

with_mask ➔ 1 (Compliant)

without_mask ➔ 0 (Non-compliant)

mask_weared_incorrect ➔ 0 (Non-compliant)

⚙️ How It Works: The ML Pipeline
1. Data Cleaning & Exploration
The script first scans the directories, extracts metadata (dimensions, file size), and filters out unreadable images or anomalies smaller than 20x20 pixels to ensure high-quality training data.

2. Image Preprocessing
Before feature extraction, each image undergoes a strict standardization process:

Grayscale Conversion: Removes color noise, focusing the model on structural shapes.

Resizing: Standardized to 128x128 pixels.

Normalization: Pixel values are scaled to a [0, 1] range.

Histogram Equalization: Enhances contrast, ensuring the model performs well regardless of the original lighting conditions.

3. Feature Extraction (HOG)
Instead of feeding raw pixels into the model, we extract structural features using HOG (Histogram of Oriented Gradients). HOG excels at capturing edge directions and local shapes (like the rigid edge of a mask or the curve of a chin), ignoring background noise.

4. Dimensionality Reduction (PCA)
HOG generates a massive feature vector. We apply PCA to compress these features down to the top 300 principal components, retaining the most critical variance while drastically reducing training time and preventing overfitting.

5. Classification (SVM)
An SVM with a Radial Basis Function (RBF) kernel is trained on the PCA-reduced features. Because the dataset becomes imbalanced when grouping without_mask and mask_weared_incorrect together, the SVM uses class_weight='balanced' to prevent bias toward the majority class.

🚀 Setup & Installation
Dependencies
Ensure you have Python installed, then install the required libraries:

Bash
pip install numpy pandas opencv-python scikit-learn scikit-image matplotlib seaborn tqdm
Configuration
Before running the script, update the DATASET_FOLDERS dictionary in the CONFIG section of the code to point to your local dataset directories.

Note: The default paths are currently set to the author's local environment:

Python
DATASET_FOLDERS = {
    'with_mask':,
    # ... update these to your local paths ...
}
📈 Outputs & Artifacts
Running the script will automatically generate and save the following artifacts in your working directory:

Visualizations
confusion_matrix.png: A heatmap showing the True Positives, True Negatives, False Positives, and False Negatives on the 20% test split.

sample_predictions.png: A 2x5 grid of random test images with their predicted labels overlaid (Green = Correct, Red = Incorrect).

Exported Models
facemask_svm_model.pkl: A comprehensive dictionary containing the trained SVM model, PCA transformer, StandardScaler, and evaluation metrics.

preprocessed_data.pkl: The transformed and split datasets (X_train, X_test, etc.) saved for fast experiment iteration without needing to re-extract HOG features.
""")

```python?code_reference&code_event_index=6
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(r"""# 😷 Face Mask Detection Using Support Vector Machines (SVM)

## 📌 Project Overview
This repository contains a complete machine learning pipeline for binary image classification to detect proper face mask usage. Instead of relying on heavy deep learning models, this project demonstrates a highly efficient, classical computer vision approach leveraging **Histogram of Oriented Gradients (HOG)** for feature extraction, **Principal Component Analysis (PCA)** for dimensionality reduction, and a **Support Vector Machine (SVM)** for robust classification.



---

## 📊 Dataset Requirements
This project utilizes the [Face Mask Detection Dataset by Vijay Kumar](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection) on Kaggle.

### Data Structure
To run the training script, download the dataset and ensure the extracted folders match the following structure:
Label Mapping (Strict Compliance)
The system is designed for strict safety compliance. Therefore, incorrectly worn masks are penalized and classified identically to having no mask at all:

with_mask ➔ 1 (Compliant)

without_mask ➔ 0 (Non-compliant)

mask_weared_incorrect ➔ 0 (Non-compliant)

⚙️ How It Works: The ML Pipeline
1. Data Cleaning & Exploration
The script first scans the directories, extracts metadata (dimensions, file size), and filters out unreadable images or anomalies smaller than 20x20 pixels to ensure high-quality training data.

2. Image Preprocessing
Before feature extraction, each image undergoes a strict standardization process:

Grayscale Conversion: Removes color noise, focusing the model on structural shapes.

Resizing: Standardized to 128x128 pixels.

Normalization: Pixel values are scaled to a [0, 1] range.

Histogram Equalization: Enhances contrast, ensuring the model performs well regardless of the original lighting conditions.

3. Feature Extraction (HOG)
Instead of feeding raw pixels into the model, we extract structural features using HOG (Histogram of Oriented Gradients). HOG excels at capturing edge directions and local shapes (like the rigid edge of a mask or the curve of a chin), ignoring background noise.

4. Dimensionality Reduction (PCA)
HOG generates a massive feature vector. We apply PCA to compress these features down to the top 300 principal components, retaining the most critical variance while drastically reducing training time and preventing overfitting.

5. Classification (SVM)
An SVM with a Radial Basis Function (RBF) kernel is trained on the PCA-reduced features. Because the dataset becomes imbalanced when grouping without_mask and mask_weared_incorrect together, the SVM uses class_weight='balanced' to prevent bias toward the majority class.

🚀 Setup & Installation
Dependencies
Ensure you have Python installed, then install the required libraries:

Bash
pip install numpy pandas opencv-python scikit-learn scikit-image matplotlib seaborn tqdm
Configuration
Before running the script, update the DATASET_FOLDERS dictionary in the CONFIG section of the code to point to your local dataset directories.

Note: The default paths are currently set to the author's local environment:

Python
DATASET_FOLDERS = {
    'with_mask': r"C:\Users\David\Desktop\Neural Network\midterm 2\Dataset\with_mask",
    # ... update these to your local paths ...
}
📈 Outputs & Artifacts
Running the script will automatically generate and save the following artifacts in your working directory:

Visualizations
confusion_matrix.png: A heatmap showing the True Positives, True Negatives, False Positives, and False Negatives on the 20% test split.

sample_predictions.png: A 2x5 grid of random test images with their predicted labels overlaid (Green = Correct, Red = Incorrect).

Exported Models
facemask_svm_model.pkl: A comprehensive dictionary containing the trained SVM model, PCA transformer, StandardScaler, and evaluation metrics.

preprocessed_data.pkl: The transformed and split datasets (X_train, X_test, etc.) saved for fast experiment iteration without needing to re-extract HOG features.
""")
print("README generated successfully.")
