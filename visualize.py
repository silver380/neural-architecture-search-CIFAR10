import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from numpy import loadtxt


def avg_fitness():
    df = pd.read_csv('histories.csv', header=None)
    df.loc['mean'] = df.mean()
    df.loc['min'] = df.min()
    df.loc['max'] = df.max()
    print(df.loc['mean'].max())
    fig, ax = plt.subplots(figsize =(12,8))
    ax.plot(df.columns,df.loc['mean'])
    ax.fill_between(df.columns, df.loc['min'], df.loc['max'], alpha=0.2)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average Fitness')
    plt.xticks(np.arange(min(df.columns), max(df.columns)+10, 11.0))
    plt.show()


if __name__ == '__main__':
    avg_fitness()