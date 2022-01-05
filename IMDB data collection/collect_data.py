'''
The purpose of this file is to fetch attributes like director, cast, budget, box office and reviews from the IMDB API. 
Director, cast, budget, box office is saved in a csv file along with imdb_id.
Reviews of a movie are stored in the directory review_data as <imdb_id>.csv for every movie in the dataset.
'''

import pandas as pd
import numpy as np
from imdb import IMDb
import re
ia = IMDb()

'''
Author: Tanmay Sawaji, Shwetha Panampilly
'''

def read_data():
    '''
    This function reads a csv file containing imdb_ids and returns it in a dataframe.
    '''
    df = pd.read_csv('links.csv')
    to_drop = ['movieId', 'tmdbId']
    df.drop(to_drop, axis = 1, inplace = True)
    df.reset_index()
    return df

def post_api_call(imdb_id):
    '''
    This function calls the IMDB API and fetches values for director, cast, budget, box office and reviews adn returns these values.
    '''
    try:
        movie_dict = ia.get_movie(imdb_id)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, False
    try:
        ia.update(movie_dict, ['reviews'])
        reviews = []
        for elem in movie_dict.get('reviews'):
            reviews.append(elem['content'])
    except Exception:
        reviews = False
    try:
        director = ""
        for elem in movie_dict['director']:
            director += str(elem) + ";"
        director = director.rstrip(';')
    except Exception:
        director = np.nan
    try:
        actors = ""
        for elem in movie_dict['cast']:
            actors += str(elem) + ";"
        actors = actors.rstrip(';')
    except Exception:
        actors = np.nan
    try:
        box_office = movie_dict['box office']['Cumulative Worldwide Gross']
        box_office = re.sub(r"[^0-9]+", '', box_office)
    except Exception:
        box_office = np.nan
    try:
        budget = movie_dict['box office']['Budget']
        budget = re.sub(r"[^0-9]+", '', budget)
    except Exception:
        budget = np.nan
    
    return director, actors, box_office, budget, reviews

def create_new_dataset(df):
    '''
    This function adds columns to the dataframe to accomodate director, cast, budget and box office. It also creates a separate csv file to store reviews.
    '''
    df['budget'] = ['' for x in range(len(df))]
    df['box_office'] = ['' for x in range(len(df))]
    df['director'] = ['' for x in range(len(df))]
    df['actors'] = ['' for x in range(len(df))]

    for i in range(len(df)):
        print("Processing movie : {}".format(i + 1))
        director, actor, box_office, budget, reviews = post_api_call(df.at[i, 'imdbId'])
        df.at[i, 'budget'] = budget
        df.at[i, 'box_office'] = box_office
        df.at[i, 'director'] = director
        df.at[i, 'actors'] = actor
        if reviews:
            new_df = pd.DataFrame({'reviews' : reviews})
            new_df.to_csv("review_data/{}.csv".format(str(df.at[i, 'imdbId'])), index = False)

    df.to_csv("data_without_actors.csv", index = False) 

def main():
    df = read_data()
    create_new_dataset(df)

if __name__ == "__main__":
    main()
