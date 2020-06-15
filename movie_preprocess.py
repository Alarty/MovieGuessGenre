import pandas as pd
import csv
from tqdm import tqdm
import json
import re
import nltk
from nltk.corpus import stopwords


def list_genres(movies):
    # list all genres
    all_genres = sum(movies['genre'], [])
    len(set(all_genres))
    all_genres = nltk.FreqDist(all_genres)
    all_genres = pd.DataFrame({'Genre': list(all_genres.keys()),
                               'Count': list(all_genres.values())})
    return all_genres


def get_input_data(folder=''):
    # read input data
    
    print("Folder :" + folder)
    print(folder + "data/movie.metadata.tsv")
    meta = pd.read_csv(folder + "data/movie.metadata.tsv", sep='\t', header=None)
    meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
    meta.head()

    plots = []
    with open(folder + "data/plot_summaries.txt", 'r', encoding="utf8") as f:
        reader = csv.reader(f, dialect='excel-tab')
        for row in tqdm(reader):
            plots.append(row)

    movie_id = []
    plot = []
    # extract movie Ids and plot summaries
    for i in tqdm(plots):
        movie_id.append(i[0])
        plot.append(i[1])

    # create dataframe
    movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})
    # change datatype of 'movie_id'
    meta['movie_id'] = meta['movie_id'].astype(str)

    # merge meta with movies
    movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on='movie_id')

    genres = []
    # extract genres
    for i in movies['genre']:
        genres.append(list(json.loads(i).values()))

    # add to 'movies' dataframe
    movies['genre'] = genres

    # remove movies without genres
    movies = movies[~(movies['genre'].str.len() == 0)]
    return movies


def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()
    return text


def remove_stopwords(text):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def trim_genres(movies, genres, nb_genres):
    list_top_genres = genres.nlargest(columns="Count", n=nb_genres)['Genre'].values
    # remove genre from movies if genre not in top
    movies['genre'] = pd.Series([[genre for genre in list_genre if genre in list_top_genres] for list_genre in movies['genre']])
    # remove movies without genres
    movies = movies[~(movies['genre'].str.len() == 0)]
    return movies[~(movies['genre'].isnull())]
    