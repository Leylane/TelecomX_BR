# EDA functions

import seaborn as sns
import matplotlib.pyplot as plt

def plot_churn_distribution(df):
    sns.countplot(x='Churn', data=df)
    plt.show()
