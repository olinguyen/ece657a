import h5py
import numpy as np
import scipy.io
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

if __name__ == "__main__":
    dataA = scipy.io.loadmat('./DataA.mat')
    print('Succesfully loaded data!')
    print("Data A shape:", dataA['fea'].shape)

    a_features = dataA['fea']

    # Replace NaN with mean
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    a_features_clean = imp.fit_transform(a_features)
    plt.plot(a_features_clean);
    plt.title("Raw data")
    plt.show()

    # Apply median filtering to remove outliers
    filtered = signal.medfilt(a_features_clean)
    plt.plot(filtered);
    plt.title("Median filtered data")
    plt.show()

    # Min-max and z-score normalization
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(filtered)
    X_train_standardized = scale(X_train_minmax)

    # Plot histograms
    plt.figure(1, figsize=(10, 6))
    plt.subplot(222).set_title("Feature 9 histogram after normalization")
    plt.hist(X_train_standardized[:, 8], bins=100);
    plt.subplot(224).set_title("Feature 24 histogram after normalization")
    plt.hist(X_train_standardized[:, 23], bins=100);

    plt.subplot(221).set_title("Feature 9 histogram before normalization")
    plt.hist(a_features_clean[:, 8], bins=100);
    plt.subplot(223).set_title("Feature 24 histogram before normalization")
    plt.hist(a_features_clean[:, 23], bins=100);
    plt.show()

    # Plot auto-correlation
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0][0].set_title("Feature 9 autocorrelation before normalization")
    axes[0][1].set_title("Feature 24 autocorrelation before normalization")
    axes[1][0].set_title("Feature 9 autocorrelation after normalization")
    axes[1][1].set_title("Feature 24 autocorrelation after normalization")

    pd.tools.plotting.autocorrelation_plot(X_train_standardized[:, 8], ax=axes[0][1])
    pd.tools.plotting.autocorrelation_plot(X_train_standardized[:, 23], ax=axes[1][1])
    pd.tools.plotting.autocorrelation_plot(a_features_clean[:, 8], ax=axes[0][0])
    pd.tools.plotting.autocorrelation_plot(a_features_clean[:, 23], ax=axes[1][0])
    plt.show()
