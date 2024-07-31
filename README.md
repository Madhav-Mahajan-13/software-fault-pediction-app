# Software Fault Prediction App

## Overview
The Data Analysis App is a powerful GUI tool designed for data scientists, analysts, and machine learning enthusiasts. It simplifies complex data analysis workflows by providing an intuitive interface for data preprocessing, sampling, and machine learning tasks.

## Features
- CSV file upload for training data
- Automated data preprocessing and missing value handling
- Multiple sampling methods for imbalanced datasets
- Wide range of machine learning models:
  - CNN + LSTM
  - Random Forest + Gradient Boosting
  - K-Means Clustering + PCA
  - Autoencoder + SVM
  - Decision Tree + Logistic Regression
  - Word2Vec + RNN
  - Naive Bayes + k-NN
  - Reinforcement Learning + Deep Q-Learning
  - Gradient Boosting + Neural Network
  - Genetic Algorithm + Neural Network
- Model training with performance metrics
- Feature importance visualization
- Model saving and loading
- Fault prediction on new datasets

## Requirements
- Python 3.x
- tkinter
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- tensorflow
- joblib

## Installation
1. Clone the repository:
2. Navigate to the project directory:
3. Install required packages:

## Usage
1. Run the application:
2. Use the GUI to:
- Upload training CSV files
- Select sampling methods
- Choose and train machine learning models
- View model performance and feature importance
- Save trained models
- Load new datasets and predict faults

## Key Components
- `DataAnalysisApp`: Main class for GUI and workflow management
- Data processing functions: `process_data()`, `handle_missing_values()`
- Model training: `train_model()`
- Prediction: `predict_faults()`

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements or bug reports.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Thanks to the open-source community and libraries like scikit-learn, TensorFlow, and imbalanced-learn.

## Disclaimer
This app is for educational and research purposes. Always validate results and consult domain experts before using in production environments.
