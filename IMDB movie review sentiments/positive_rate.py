'''
Author : Tanmay Sawaji
This reads review data for every movieId in the dataset and calls the Naive Bayes classifier to classify the review.
Then it adds a new column to the main dataset and stores the percentage of positive reviews of a movie.
'''

import pickle
import pandas as pd
from NaiveBayesClassifier import NaiveBayes

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def read_dataset(path):
    df = pd.read_csv(path)
    return df

def read_reviews(filename):
    path = "../IMDB data collection/review_data/" + filename + ".csv"
    examples = []
    with open(path, 'r', encoding='utf8') as f:
        for index,line in enumerate(f):
            if index == 0:
                continue
            line = line.strip()
            example = line.replace("<br />", "") if len(line) > 1 else ""
            examples.append(example)
    return {"examples" : examples}

def main():
    path = "../Datasets/MovieDataset_P556.csv"
    df = read_dataset(path)
    model = load_model("50kmodel.pkl")
    positive_rate = []
    for index in range(df.shape[0]):
        print(f"Processing entry {index + 1} / {df.shape[0]}")
        imdb_id = df.at[index, "imdbId"]
        test_cases = read_reviews(str(imdb_id))
        results = model.fit_naive_bayes(test_cases, "examples")
        pos_count = sum([ (x == "positive") for _,x in enumerate(results)])
        pos_val = pos_count * 100 / len(results)
        positive_rate.append(pos_val)
        print(f"The positive rate is {pos_val}")
    
    df["positive_rate"] = positive_rate

    df.to_csv("data_with_pos_rate.csv", index = False)
    

if __name__ == "__main__":
    main()