# Twitter Bot Detection - Kaggle Competition

This repository contains my solution to the **Twitter Bot Detection** Kaggle competition. The goal of the competition is to classify Twitter accounts as bots or genuine users based on their profile attributes and activity metrics. Below, I outline the steps taken to preprocess the data, engineer features, build models, and optimize performance.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Model Optimization](#model-optimization)
7. [Evaluation Metrics](#evaluation-metrics)
8. [How to Run the Code](#how-to-run-the-code)
9. [Tools and Libraries](#tools-and-libraries)

---

## Project Overview

The project involves identifying bot accounts on Twitter by analyzing metadata and text features from their profiles. The solution incorporates advanced feature engineering, ensemble modeling, and hyperparameter optimization using **Optuna**.

---

## Dataset Description

The dataset contains the following key attributes for each Twitter account:
- **Profile attributes:** `description`, `created_at`, `location`, etc.
- **Activity metrics:** `followers_count`, `friends_count`, `statuses_count`, etc.
- **Target:** Binary classification label indicating whether the account is a bot (`1`) or genuine (`0`).

---

## Preprocessing Steps

- **Missing Values Handling**: Filled missing values in `description`, `location`, and `lang` with placeholders like "Unknown."
- **Binary Column Encoding**: Converted `True`/`False` values to binary (`0`/`1`).
- **Datetime Features**: Extracted `year`, `month`, `day_of_week`, and `hour` from the `created_at` column.
- **Scaling**: Applied scaling using `StandardScaler` to normalize numerical features.

---

## Feature Engineering

Key features engineered for better model performance:
1. **Text-Based Features**:
   - Length of `description`, number of words, and average word length.
   - Sentiment polarity and subjectivity using `TextBlob`.
   - Presence of URLs, email addresses, emojis, and hashtags in `description`.

2. **Interaction Metrics**:
   - Ratios such as `followers_to_friends`, `followers_per_day`, and `engagement_ratio`.
   - Clustering of descriptions using TF-IDF and KMeans.

3. **Profile Customization Indicators**:
   - Binary flags for `has_custom_profile_image` and `has_custom_background`.

4. **One-Hot Encoding**:
   - Categorical features like `lang` and top locations were one-hot encoded.

5. **Outlier Handling**:
   - Outliers in numerical columns were clipped using IQR-based thresholds.

---

## Model Training

### **Base Models**
1. **Logistic Regression**
2. **XGBoost**
3. **LightGBM**
4. **Neural Networks**

### **Ensemble Model**
- Combined Neural Networks and LightGBM using a soft-voting ensemble for better generalization.

---

## Model Optimization

- **Hyperparameter Tuning**: Used **Optuna** to optimize the following:
  - **Neural Networks**: Number of units, dropout rates, learning rates, etc.
  - **LightGBM**: Number of estimators, max depth, learning rates, etc.

- **Cross-Validation**: Implemented 5-fold stratified cross-validation to ensure robust evaluation.

---

## Evaluation Metrics

- **AUC-ROC**: Primary metric used to evaluate model performance.

---

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/harshvs4/twitter-bot-detection.git
   cd twitter-bot-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Tools and Libraries

- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `xgboost`, `lightgbm`, `tensorflow`, `textblob`, `emoji`, `optuna`.
- **Visualization**: `matplotlib`, `seaborn`, `SweetViz`, `WordCloud`.
- **Optimization**: `Optuna` for hyperparameter tuning.

---

## Future Work

- **Additional Features**: Incorporate NLP embeddings for better text representation.
- **Advanced Models**: Explore transformer-based architectures like BERT for text-based features.
- **Automation**: Integrate data pipelines for seamless preprocessing and feature generation.

---

Feel free to explore the repository and provide feedback or suggestions! 🚀