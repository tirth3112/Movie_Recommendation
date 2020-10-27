import pandas as pd
import numpy as np

movies_df = pd.read_csv('bolly_movies.csv',usecols=['movie_id','title'],dtype={'movie_id': 'int32', 'title': 'str'})
rating_df=pd.read_csv('bolly_ratings.csv',usecols=['user_id', 'movie_id', 'rating'],
    dtype={'user_id': 'int32', 'movieId': 'int32', 'rating': 'float32'})
rating_df.head()
df = pd.merge(rating_df,movies_df,on='movie_id')
df.head()
combine_movie_rating = df.dropna(axis = 0, subset = ['title'])
movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )

rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(movie_ratingCount['totalRatingCount'].describe())
popularity_threshold = 30
rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_movie.head()
movie_features_df=rating_popular_movie.pivot_table(index=['title'],columns='user_id',values='rating').fillna(0)
#print(movie_features_df.values)
from scipy.sparse import csr_matrix
movie_features_df_matrix = csr_matrix(movie_features_df.values)
print(movie_features_df_matrix)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)

name=input("ENter Mivie Name")
movie_name_list= movie_features_df.index.tolist()
movie_name_list = list(movie_name_list)
movie_name_index = movie_name_list.index(name)

distances, indices = model_knn.kneighbors(movie_features_df.iloc[movie_name_index,:].values.reshape(1, -1), n_neighbors = 4)


# %%
movie_features_df.head()


# %%
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[movie_name_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))

