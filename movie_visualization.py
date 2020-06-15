import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns


def plot_genres(all_genres):
    g = all_genres.nlargest(columns="Count", n=50)
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=g, x="Count", y="Genre")
    ax.set(ylabel='Count')
    plt.show()


def plot_freq_words(data, terms = 30, title=None):
    all_words = ' '.join([text for text in data])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms)

    # visualize words and frequencies
    plt.figure()
    ax = sns.barplot(data=d, x="count", y="word")
    ax.set(ylabel='Word')
    if title:
        plt.title(title)
    plt.show()
