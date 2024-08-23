# Music Genre Prediction Portfolio

Welcome to my portfolio! This repository showcases a Jupyter notebook that predicts music genre based on age and gender using machine learning techniques with Python. Below you'll find a summary of the project and the key steps involved.

## Project Overview

This project uses a dataset containing information about individuals, including their age, gender, and the music genre they prefer. The goal is to build a machine learning model to predict music genre based on age and gender.

### Technologies Used

- **Python**: Programming language used for data manipulation and model building.
- **Pandas**: Library for data handling and preprocessing.
- **Scikit-Learn**: Library for building and evaluating the machine learning model.
- **Joblib**: Library for saving and loading the model.
- **Graphviz**: Tool for visualizing the decision tree.

## Key Steps

1. **Exploratory Data Analysis (EDA)**: Understanding the dataset and its features.
2. **Data Preparation**: Splitting the data into input features and target labels.
3. **Model Building**: Using `DecisionTreeClassifier` from Scikit-Learn to create the model.
4. **Training and Testing**: Training the model with 80% of the data and testing it with 20% to evaluate performance.
5. **Model Persistence**: Saving the trained model using `joblib` for future use.
6. **Model Visualization**: Exporting and visualizing the decision tree to understand the model's decision-making process.

## Usage

1. **Data Loading**: The dataset is loaded from a CSV file.
   
   ```python
   music_data = pd.read_csv('/kaggle/input/music-data/music.csv')
   ```
   
