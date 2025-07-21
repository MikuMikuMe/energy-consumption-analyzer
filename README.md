# energy-consumption-analyzer

To create a comprehensive tool that analyzes energy consumption patterns using machine learning, we can follow the process of loading data, preprocessing it, selecting and training a machine learning model, and finally optimizing consumption patterns based on the analysis. Here's an illustrative Python program for such a project, including comments and basic error handling.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """ Load data from CSV file. """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """ Preprocess the data: handle missing values, encode categorical data, scale features. """
    try:
        # Handle missing values
        data.fillna(data.median(), inplace=True)

        # Encode categorical features if any
        for column in data.select_dtypes(include='object').columns:
            data[column] = LabelEncoder().fit_transform(data[column])

        # Feature scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        print("Data preprocessed successfully")
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return None

def split_data(data, target_column):
    """ Split the data into training and testing sets. """
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into train and test sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in splitting data: {e}")
        return None, None, None, None

def train_model(X_train, y_train):
    """ Train a Random Forest model. """
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model trained successfully")
        return model
    except Exception as e:
        print(f"Error in training model: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """ Evaluate the model using mean absolute error and mean squared error. """
    try:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model evaluation completed\nMAE: {mae}\nMSE: {mse}")
        return mae, mse
    except Exception as e:
        print(f"Error in evaluating model: {e}")
        return None, None

def analyze_energy_consumption(data, target_column='energy_usage'):
    """ Main function to load, preprocess, train, and evaluate model for energy consumption analysis. """
    data = load_data(data)
    if data is not None:
        data = preprocess_data(data)
    if data is not None:
        X_train, X_test, y_train, y_test = split_data(data, target_column)
    if X_train is not None:
        model = train_model(X_train, y_train)
    if model is not None:
        evaluate_model(model, X_test, y_test)

def visualize_data(data, target_column='energy_usage'):
    """ Visualize energy consumption data to identify patterns. """
    try:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x=data.index, y=target_column)
        plt.title('Energy Consumption Over Time')
        plt.xlabel('Time')
        plt.ylabel('Energy Usage')
        plt.show()
    except Exception as e:
        print(f"Error in visualization: {e}")

if __name__ == '__main__':
    file_path = 'energy_consumption_data.csv' # Replace with actual path
    energy_data = load_data(file_path)
    
    # If successfully loaded, visualize data
    if energy_data is not None:
        visualize_data(energy_data, target_column='energy_usage') 

    # Perform analysis
    analyze_energy_consumption(file_path, target_column='energy_usage')
```

### Key Notes:
- **Loading Data**: Use `pd.read_csv` to load the data. Ensure the file path is correct.
- **Error Handling**: Each function incorporates basic error handling using try-except blocks to catch and report exceptions gracefully.
- **Preprocessing**: Includes handling missing values, encoding categorical data, and feature scaling.
- **Model Training**: Using a `RandomForestRegressor` to predict energy consumption.
- **Visualization**: A basic plot to visualize energy consumption over time.
- **Assumptions**: Path to the CSV file and the target column name (`energy_usage`) are used. Make sure to update them according to your dataset.

This program serves as a skeleton and might need adjustments depending on specifics like feature engineering, model selection, and hyperparameter tuning based on the dataset's actual contents and requirements.