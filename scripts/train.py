import os
import numpy as np
import csv
from numpy import array
"""
- LSTM model will learn a function that maps a sequence of past observations as input to an output observation
- The sequence of observations must be transformed into multiple examples from which the LSTM can learn

"""

# can change tickers just putting it like that for now 
def load_and_preprocess_data(tickers=["MSFT", "IBM", "NVDA", "TSLA"], n_steps=3):
    """
    load data from CSV files and preprocess into sequences.
    
    Args:
        tickers (list of str): List of ticker symbols
        n_steps (int): Number of time steps to consider for each sequence.

    Returns:
        tuple: A tuple containing two elements:
            - List of numpy arrays, where each array represents input sequences for a specific ticker.
            - Numpy array of output sequences.
    """
    # Get input files for each ticker
    input_files = [f"{ticker}_monthly_data.csv" for ticker in tickers]
    X_all, y_all = [], []

    for input_file in input_files:
        with open(os.path.join("data", input_file), 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            data = list(reader)
        
        # Extracting metrics data from CSV
        metric_data = np.array(data)[:, 1:].astype(float)  # Exclude Date column and convert to float

        # Split sequence for each metric
        X_metric, y_metric = split_sequence(metric_data, n_steps)
        X_all.append(X_metric)
        y_all.append(y_metric)

    # Each ticker has its own array of input sequences
    X_combined = [np.concatenate(X_ticker, axis=1) for X_ticker in zip(*X_all)]
    y_combined = np.concatenate(y_all)
    
    return X_combined, y_combined

def split_sequence(sequence, n_steps):
    """
    Split a univariate sequence into samples.

    Args:
        sequence (numpy array): Input sequence.
        n_steps (int): Number of time steps to consider for each sequence.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - Input sequences.
            - Output sequences.
    """
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix][-1]  # Only take the last value as y
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Example usage
tickers = ["MSFT", "IBM"]
X, y = load_and_preprocess_data(tickers, n_steps=3)
print("Input sequences:")
print(X)
print("Output sequences:")
print(y)

