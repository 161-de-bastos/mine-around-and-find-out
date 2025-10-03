import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(scores_avg, title="Distribución de sentimiento (promedio corpus)"):
    labels = list(scores_avg.keys())
    vals = [scores_avg[k] for k in labels]
    
    plt.figure()
    sns.barplot(x=labels, y=vals)
    plt.title(title)
    plt.ylabel("Probabilidad media")
    plt.tight_layout()

def plot_degree_hist(deg, title="Distribución de grados"):
    plt.figure()
    sns.histplot(deg, bins=20, kde=False)
    plt.title(title)
    plt.xlabel("grado")
    plt.ylabel("frecuencia")
    plt.tight_layout()