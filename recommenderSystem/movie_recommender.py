import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# helper function
def get_title_from_index(index):
    return df[df.index == index]["movie_title"].values[0]

def get_index_from_title(title):
    return df[df.movie_title == title]["index"].values[0]

#step 1: Read CSV
df = pd.read_csv("movie_dataset.csv")
# print(df.head())

# Step 2: Select Features
features = ["director_name", "actor_2_name", "genres", "actor_3_name", "plot_keywords"]

# Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna("")

def  combine_features(row):
    try:
        return row["director_name"] +" "+row["actor_2_name"] +" "+row["genres"] +" "+ row["actor_3_name"]+ " "+ row["plot_keywords"]
    except:
        print("Error: ", row)

df["combined_features"] = df.apply(combine_features, axis = 1)
# print("Combined Features:", df["combined_features"].head())

# Step 4: Create count matrix from this combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])
# print(count_matrix)

# Step 5: Compute CosineSimilarity based on the matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "AvatarÂ"

# Step6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

# Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1],reverse=True)

# Step 8: Print titles of first 50
count = 0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    count += 1
    if count > 50:
        break