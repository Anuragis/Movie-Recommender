import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import prediction_algorithms
from surprise import accuracy, evaluate
import csv

user_ratings_df = pd.read_csv('train.csv')
test_ratings_df = pd.read_csv('test.csv')
user_ratings_df = user_ratings_df.drop(columns=['Timestamp'])

user_ratings_df = user_ratings_df.append(test_ratings_df)

movie_names = user_ratings_df.columns.values.tolist()[5:]
movies_dict = {movie: id for id, movie in enumerate(movie_names)}


movie_ratings = []
for index, row in user_ratings_df.iterrows():
    for key in movies_dict.keys():
        movie_ratings.append((row['ID'], movies_dict[key], row[key]))


movie_ratings_df = pd.DataFrame.from_records(
    data=movie_ratings, columns=['userID', 'itemID', 'rating'])

reader = Reader(rating_scale=(1, 5))
movie_ratings_df.dropna(inplace=True)

data = Dataset.load_from_df(
    movie_ratings_df[['userID', 'itemID', 'rating']], reader)

data = data.build_full_trainset()

""" sim_options = {'name': 'pearson_baseline',
               'user_based': True
               } """

algo = prediction_algorithms.SVD(n_epochs=100)
algo.fit(data)
predictions = []
print("Item Id for despicable Me ", movies_dict['Rating [Despicable Me]'])
movie_id = movies_dict['Rating [Despicable Me]']
for id in test_ratings_df['ID']:
    pred = algo.predict(iid=movie_id, uid=id)
    predictions.append((id, pred.est))

op = open("predictions.csv", "w")
wr = csv.writer(op)
wr.writerow(["ID", "Rating [Despicable Me]"])
wr.writerows(predictions)
op.close()
# Test data prepare
""" test_ratings_df = pd.read_csv('test.csv')
test_movie_ratings = []
for index, row in test_ratings_df.iterrows():
    for key in movies_dict.keys():
        test_movie_ratings.append((row['ID'], movies_dict[key], row[key]))

test_movie_ratings_df = pd.DataFrame.from_records(
    data=test_movie_ratings, columns=['ID', 'movieID', 'rating'])

test_data = Dataset.load_from_df(
    test_movie_ratings_df[['ID', 'movieID', 'rating']], reader) """
