import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


if __name__ == "__main__":
    dataB = scipy.io.loadmat('./DataB.mat')
    print('Succesfully loaded data!')
    print("Data B shape:", dataB['fea'].shape)

    b_features = dataB['fea']
    X_train = dataB['fea']
    y_train = dataB['gnd'].ravel()

    # Visualize handwritten digits
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        img = X_train[[y_train == i]][0].reshape(28, 28)
        ax[i].imshow(img, cmap='gray')
    plt.show()

    # Run PCA
    pca = PCA()
    X_transformed = pca.fit_transform(X_train)

    # Plot first and second principal component
    fig, plot = plt.subplots(figsize=(8, 6))
    #fig.set_size_inches(25, 25)
    plt.prism()

    colors = [
        ('#27ae60', 'o'),
        ('#2980b9', 'o'),
        ('#8e44ad', 'o'),
        ('#f39c12', 'o'),
        ('#c0392b', 'o'),
    ]
    labels = np.unique(y_train)
    for num in range(len(labels)):
        plt.scatter([X_transformed[:,0][i] for i in range(len(y_train)) if y_train[i] == num],
        [X_transformed[:,1][i] for i in range(len(y_train)) if y_train[i] == num], 15,
        label=str(num), color = colors[num][0], marker=colors[num][1])#, edgecolors='black')

    plt.legend(scatterpoints=1)
    plt.title("PCA Using the 1st and 2nd principal component")
    plot.set_xticks(())
    plot.set_yticks(())
    plt.tight_layout()
    plt.show()

    # Plot 5th and 6th principal component
    for num in range(len(labels)):
        plt.scatter([X_transformed[:,4][i] for i in range(len(y_train)) if y_train[i] == num],
        [X_transformed[:,5][i] for i in range(len(y_train)) if y_train[i] == num], 15,
        label=str(num), color = colors[num][0], marker=colors[num][1])#, edgecolors='black')

    plt.legend(scatterpoints=1)
    plt.title("PCA Using 5th and 6th principal component")

    plot.set_xticks(())
    plot.set_yticks(())
    plt.tight_layout()
    plt.show()

    # Classify using Naive Bayes
    gnb = GaussianNB()
    y_train = y_train.ravel()
    error_rates = []
    rm = []
    components = [2, 4, 10, 30, 60, 200, 500, 784]

    for component in components:
        X_train = X_transformed[:, :component]
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_train)
        error_rate = (y_pred != y_train).mean()
        error_rates.append(error_rate)
        rm.append(pca.explained_variance_ratio_[:component].sum())
        print("Error rate: %.6f (%d components)" % (error_rate, component))

    # Plot error rates vs retained variance
    plt.plot(rm, error_rates)
    plt.xlabel("Retained variance")
    plt.ylabel("Error rate")
    plt.title("Classification error vs retained variance")
    plt.show()

    # Plot LDA components
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_transformed = lda.fit_transform(X_train, y_train)

    for num in range(len(labels)):
        plt.scatter([X_transformed[:,0][i] for i in range(len(y_train)) if y_train[i] == num],
        [X_transformed[:,1][i] for i in range(len(y_train)) if y_train[i] == num], 15,
        label=str(num), color = colors[num][0], marker=colors[num][1])#, edgecolors='black')

    plt.legend(scatterpoints=1)
    plt.title("LDA Using the 1st and 2nd principal component")
    plot.set_xticks(())
    plot.set_yticks(())
    plt.tight_layout()
    plt.show()
