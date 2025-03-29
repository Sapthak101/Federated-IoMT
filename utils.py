import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path='synthetic_iomt_dataset.csv'):
    df = pd.read_csv(path)
    X = df.drop(['execution_state'], axis=1)
    y = df['execution_state']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def evaluate_accuracy(model, X, y):
    _, acc = model.evaluate(X, y, verbose=0)
    return acc