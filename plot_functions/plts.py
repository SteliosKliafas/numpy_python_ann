import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np


def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_heatmap(confusion):
    heat = sns.heatmap(confusion, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')


def plot_pca(train_x, train_y):
    pca = PCA(2)
    projected = pca.fit_transform(train_x)
    plt.scatter(projected[:, 0], projected[:, 1], c=train_y, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.xlabel('pca component 1')
    plt.ylabel('pca component 2')
    plt.colorbar()


def plot_bar_chart(labels, values):
    plt.bar(labels, values, 1, edgecolor='y')
    plt.xticks(labels)
    plt.ylabel("Data Number")
    plt.xlabel("Class Labels")
    plt.show()