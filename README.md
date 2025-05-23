# Software Fault Prediction System

A machine learning-based application for predicting fault-prone software modules using code metrics.

## Overview

This application allows software quality engineers and developers to analyze code metrics data and predict which modules are likely to contain faults. The system supports multiple dataset formats, provides various sampling techniques to handle class imbalance, and includes several machine learning models for comparison.

## Features

- **Multiple Dataset Support**: Works with various software metrics datasets (NASA MDP, PROMISE repositories, etc.)
- **Comprehensive Preprocessing Pipeline**: Handles missing values, different column structures, and feature scaling
- **Class Imbalance Handling**: Provides multiple sampling techniques including SMOTE, ADASYN, and undersampling
- **Multiple ML Models**: Includes ensemble methods, traditional classifiers, and neural networks
- **Model Comparison**: Compare different models based on accuracy, precision, recall, and F1-score
- **User-Friendly Interface**: Simple GUI built with Tkinter for easy operation

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - scikit-learn
  - imbalanced-learn
  - tkinter
  - numpy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/software-fault-prediction.git
cd software-fault-prediction

# Install required packages
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Through the GUI:
   - Select training CSV files using the "Browse" button
   - Select a testing CSV file
   - Choose whether to perform sampling if data is imbalanced
   - Select a sampling method if applicable
   - Choose a model or models to train
   - Click "Train Model" or "Compare Models"

## Supported Datasets

The system works with various software metrics datasets including:

- **Object-Oriented Metrics**: ant, arc, camel, poi, tomcat (21 metrics)
- **NASA Metrics**: CM1, MW1, PC1, PC3, PC4 (38 metrics)
- **PROMISE Metrics**: ar1-ar6 (30 metrics)

## Supported Models

- Random Forest + Gradient Boosting
- K-Means Clustering + Principal Component Analysis (PCA)
- Support Vector Machine (SVM)
- Decision Tree + Logistic Regression
- Naive Bayes + k-Nearest Neighbors (k-NN)
- Neural Network

## Sample Workflow

1. **Data Selection**: Choose compatible datasets for training and testing
2. **Sampling**: Apply SMOTE to address class imbalance (typically 90%+ non-faulty modules)
3. **Model Selection**: Try different models to find the best performer for your data
4. **Evaluation**: Compare models using precision, recall, and F1-score
5. **Result Analysis**: Identify the best model for your specific project context

## Recommended Dataset Combinations

For best results, use datasets with similar structures:

- **Object-oriented metrics**: ant1.3.csv, camel1.0.csv, arc.csv, tomcat.csv
- **NASA metrics**: CM1.csv, PC1.csv, PC3.csv, PC4.csv
- **AR metrics**: ar1.csv, ar3.csv, ar4.csv, ar5.csv, ar6.csv

## Common Issues and Solutions

- **SettingWithCopyWarning**: The system handles DataFrame operations properly to avoid pandas warnings
- **Sampling Errors**: For datasets with very few fault examples, the system automatically adjusts sampling parameters
- **Column Mismatches**: The system handles different column structures and naming conventions across datasets

## Future Improvements

- Support for deep learning models
- Feature importance visualization
- Integration with CI/CD pipelines
- Defect density prediction (not just binary classification)
- Cross-project prediction capabilities

## License

[MIT License](LICENSE)

## Acknowledgments

- NASA Metrics Data Program
- PROMISE Software Engineering Repository
- The scikit-learn and imbalanced-learn communities# Software Fault Prediction System

A machine learning-based application for predicting fault-prone software modules using code metrics.

## Overview

This application allows software quality engineers and developers to analyze code metrics data and predict which modules are likely to contain faults. The system supports multiple dataset formats, provides various sampling techniques to handle class imbalance, and includes several machine learning models for comparison.

## Features

- **Multiple Dataset Support**: Works with various software metrics datasets (NASA MDP, PROMISE repositories, etc.)
- **Comprehensive Preprocessing Pipeline**: Handles missing values, different column structures, and feature scaling
- **Class Imbalance Handling**: Provides multiple sampling techniques including SMOTE, ADASYN, and undersampling
- **Multiple ML Models**: Includes ensemble methods, traditional classifiers, and neural networks
- **Model Comparison**: Compare different models based on accuracy, precision, recall, and F1-score
- **User-Friendly Interface**: Simple GUI built with Tkinter for easy operation

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - scikit-learn
  - imbalanced-learn
  - tkinter
  - numpy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/software-fault-prediction.git
cd software-fault-prediction

# Install required packages
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Through the GUI:
   - Select training CSV files using the "Browse" button
   - Select a testing CSV file
   - Choose whether to perform sampling if data is imbalanced
   - Select a sampling method if applicable
   - Choose a model or models to train
   - Click "Train Model" or "Compare Models"

## Supported Datasets

The system works with various software metrics datasets including:

- **Object-Oriented Metrics**: ant, arc, camel, poi, tomcat (21 metrics)
- **NASA Metrics**: CM1, MW1, PC1, PC3, PC4 (38 metrics)
- **PROMISE Metrics**: ar1-ar6 (30 metrics)

## Supported Models

- Random Forest + Gradient Boosting
- K-Means Clustering + Principal Component Analysis (PCA)
- Support Vector Machine (SVM)
- Decision Tree + Logistic Regression
- Naive Bayes + k-Nearest Neighbors (k-NN)
- Neural Network

## Sample Workflow

1. **Data Selection**: Choose compatible datasets for training and testing
2. **Sampling**: Apply SMOTE to address class imbalance (typically 90%+ non-faulty modules)
3. **Model Selection**: Try different models to find the best performer for your data
4. **Evaluation**: Compare models using precision, recall, and F1-score
5. **Result Analysis**: Identify the best model for your specific project context

## Recommended Dataset Combinations

For best results, use datasets with similar structures:

- **Object-oriented metrics**: ant1.3.csv, camel1.0.csv, arc.csv, tomcat.csv
- **NASA metrics**: CM1.csv, PC1.csv, PC3.csv, PC4.csv
- **AR metrics**: ar1.csv, ar3.csv, ar4.csv, ar5.csv, ar6.csv

## Common Issues and Solutions

- **SettingWithCopyWarning**: The system handles DataFrame operations properly to avoid pandas warnings
- **Sampling Errors**: For datasets with very few fault examples, the system automatically adjusts sampling parameters
- **Column Mismatches**: The system handles different column structures and naming conventions across datasets

## Future Improvements

- Support for deep learning models
- Feature importance visualization
- Integration with CI/CD pipelines
- Defect density prediction (not just binary classification)
- Cross-project prediction capabilities

## License

[MIT License](LICENSE)

## Acknowledgments

- NASA Metrics Data Program
- PROMISE Software Engineering Repository
- The scikit-learn and imbalanced-learn communities