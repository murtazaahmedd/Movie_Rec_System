from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('movie_recommendation_model.h5')

# Load data (adjust paths as needed)
movies = pd.read_csv('dataset/movies.csv')
ratings = pd.read_csv('dataset/ratings.csv')

# Preprocess data
user_ids = ratings["userId"].unique().tolist()
userencoded = {x: i for i, x in enumerate(user_ids)}
user_rev = {i: x for i, x in enumerate(user_ids)}

movie_ids = ratings['movieId'].unique().tolist()
moviecoded = {x: i for i, x in enumerate(movie_ids)}
movie_rev = {i: x for i, x in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(userencoded)
ratings['movie'] = ratings['movieId'].map(moviecoded)
ratings['rating'] = (ratings['rating'] - ratings['rating'].mean()) / ratings['rating'].std()
max_rating = max(ratings['rating'])
min_rating = min(ratings['rating'])
ratings['rating'] = ratings['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))

# Extract unique genres
genres = set()
for genre_list in movies['genres']:
    genres.update(genre_list.split('|'))
genres = list(genres)

# Create genre columns
for genre in genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

# Recommendation function for multiple genres
def recommend_movies(user_id, selected_genres, top_n=10):
    if user_id not in userencoded:
        return {"error": f"User ID {user_id} not found."}

    user_encoder = userencoded[user_id]
    movies_watched = ratings[ratings['user'] == user_encoder][['movieId', 'rating']]
    liked_movies = movies_watched[movies_watched['rating'] > 0.8]

    # Movies that the user has not watched
    movies_not_watched = movies[~movies["movieId"].isin(movies_watched['movieId'])]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(moviecoded.keys())))
    
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), [[moviecoded[x]] for x in movies_not_watched]))

    predicted_ratings = model.predict([user_movie_array[:, 0], user_movie_array[:, 1]]).flatten()

    # Handle both single and multiple genre selections
    if isinstance(selected_genres, list):  # Multiple genres selected
        filtered_movies = movies[movies[selected_genres].sum(axis=1) > 0]
    else:  # Single genre selected
        filtered_movies = movies[movies[selected_genres] == 1]

    filtered_movie_ids = filtered_movies['movieId'].values
    filtered_predicted_ratings = {}

    for movie_id in filtered_movie_ids:
        if movie_id in moviecoded:
            index = moviecoded[movie_id]
            if index < len(predicted_ratings):
                filtered_predicted_ratings[movie_id] = predicted_ratings[index]

    # Sort by predicted ratings and get top N
    top_movies = sorted(filtered_predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommended = [{"title": movies[movies["movieId"] == movie_id]["title"].values[0],
                    "genres": movies[movies["movieId"] == movie_id]["genres"].values[0]}
                   for movie_id, _ in top_movies]

    return recommended


# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Load available genres for the dropdown
    genre_list = [genre for genre in genres]
    return render_template('index.html', genres=genre_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        selected_genres = request.form.getlist('genres')  # Get selected genres
        top_n = int(request.form['top_n'])

        # Call recommend_movies function with the selected genres
        recommended_movies = recommend_movies(user_id, selected_genres, top_n)

        if 'error' in recommended_movies:
            return jsonify({'error': recommended_movies['error']})

        return render_template('index.html', genres=genres, recommendations=recommended_movies)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
