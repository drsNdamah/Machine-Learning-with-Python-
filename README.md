# Machine Learning - Music Genre Prediction Project

Welcome to my portfolio! This repository showcases a Jupyter notebook that predicts music genre based on age and gender using machine learning techniques with Python. Below you'll find a summary of the project and the key steps involved.

## Project Overview

This project uses a dataset containing information about individuals, including their age, gender, and the music genre they prefer. The goal is to build a machine learning model to predict music genre based on age and gender.

### Technologies Used

- **Python**: Programming language used for data manipulation and model building.
- **Pandas**: Library for data handling and preprocessing.
- **Scikit-Learn**: Library for building and evaluating the machine learning model.
- **Joblib**: Library for saving and loading the model.
- **Graphviz**: Tool for visualizing the decision tree.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

```


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

2. **Splitting the Data**: The dataset is divided into training and testing sets.

   ```python
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   ```

3. **Training the Model**: A decision tree classifier is trained.
   
   ```python
   predictions = model.predict(X_test)
   score = accuracy_score(y_test, predictions)
   ```
   
4. **Making Predictions**: Predictions are made and accuracy is calculated.
   
   ```python
   
   predictions = model.predict(X_test)
   score = accuracy_score(y_test, predictions)
   
   ```
   
5. **Saving the Model**: The model is saved for future use.
   
   ```python
   joblib.dump(model, 'music-recommender.joblib')
   
   ```

6. **Loading and Using the Model**: The saved model is loaded and used for making new predictions.
   
   ```python
   model = joblib.load('music-recommender.joblib')
   predictions = model.predict([[21, 1]])
   ```
   

7. **Visualizing the Decision Tree**: The decision tree is visualized and saved as a DOT file.

   ```python
   tree.export_graphviz(model, out_file='music-recommender.dot', 
                     feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
   ```

8. **Visual Representation of the Model**:

![Graph_Viz](https://github.com/user-attachments/assets/bc06d6b3-6edf-4bb7-91a6-e101035986a6)


9. **Files**

* **music-recommender.joblib**: Saved machine learning model.
* **music-recommender.dot**: Visualization of the decision tree.

