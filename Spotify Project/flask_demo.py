from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Create sample data
def create_sample_data():
    """Create sample data for demonstration"""
    # Define genres and their characteristics
    genres = {
        'pop': {'danceability': 0.7, 'energy': 0.6, 'valence': 0.7, 'tempo': 120},
        'rock': {'danceability': 0.5, 'energy': 0.8, 'valence': 0.6, 'tempo': 130},
        'hip_hop': {'danceability': 0.8, 'energy': 0.7, 'valence': 0.5, 'tempo': 90},
        'electronic': {'danceability': 0.9, 'energy': 0.9, 'valence': 0.6, 'tempo': 128},
        'jazz': {'danceability': 0.4, 'energy': 0.4, 'valence': 0.5, 'tempo': 100},
        'classical': {'danceability': 0.2, 'energy': 0.3, 'valence': 0.4, 'tempo': 80},
        'country': {'danceability': 0.6, 'energy': 0.5, 'valence': 0.7, 'tempo': 110},
        'r&b': {'danceability': 0.7, 'energy': 0.5, 'valence': 0.6, 'tempo': 95}
    }
    
    # Sample artists and tracks
    artists = {
        'pop': ['Taylor Swift', 'Ed Sheeran', 'Ariana Grande', 'Justin Bieber', 'Billie Eilish'],
        'rock': ['Queen', 'Led Zeppelin', 'The Beatles', 'AC/DC', 'Guns N Roses'],
        'hip_hop': ['Drake', 'Kendrick Lamar', 'Travis Scott', 'Post Malone', 'J. Cole'],
        'electronic': ['Daft Punk', 'Skrillex', 'Deadmau5', 'Calvin Harris', 'Martin Garrix'],
        'jazz': ['Miles Davis', 'John Coltrane', 'Louis Armstrong', 'Ella Fitzgerald', 'Duke Ellington'],
        'classical': ['Mozart', 'Beethoven', 'Bach', 'Chopin', 'Tchaikovsky'],
        'country': ['Johnny Cash', 'Dolly Parton', 'Willie Nelson', 'Taylor Swift', 'Luke Combs'],
        'r&b': ['Beyoncé', 'Rihanna', 'Usher', 'Alicia Keys', 'John Legend']
    }
    
    tracks = {
        'pop': ['Shape of You', 'Blinding Lights', 'Watermelon Sugar', 'Levitating', 'Good 4 U'],
        'rock': ['Bohemian Rhapsody', 'Stairway to Heaven', 'Hey Jude', 'Thunderstruck', 'Sweet Child O Mine'],
        'hip_hop': ['God\'s Plan', 'HUMBLE.', 'SICKO MODE', 'Circles', 'No Role Modelz'],
        'electronic': ['One More Time', 'Bangarang', 'Ghosts \'n\' Stuff', 'This Is What You Came For', 'Animals'],
        'jazz': ['Kind of Blue', 'Giant Steps', 'What a Wonderful World', 'Summertime', 'Take the A Train'],
        'classical': ['Eine kleine Nachtmusik', 'Symphony No. 9', 'Brandenburg Concerto', 'Nocturne', 'Swan Lake'],
        'country': ['Ring of Fire', 'Jolene', 'On the Road Again', 'Love Story', 'Beautiful Crazy'],
        'r&b': ['Single Ladies', 'Umbrella', 'Yeah!', 'Fallin\'', 'All of Me']
    }
    
    # Create sample data
    data = []
    track_id = 0
    
    for genre, characteristics in genres.items():
        genre_artists = artists[genre]
        genre_tracks = tracks[genre]
        
        for i in range(20):  # 20 tracks per genre
            artist = np.random.choice(genre_artists)
            track = np.random.choice(genre_tracks)
            
            # Generate audio features based on genre characteristics with some randomness
            danceability = np.random.normal(characteristics['danceability'], 0.1)
            energy = np.random.normal(characteristics['energy'], 0.1)
            valence = np.random.normal(characteristics['valence'], 0.1)
            tempo = np.random.normal(characteristics['tempo'], 10)
            
            # Ensure values are within valid ranges
            danceability = np.clip(danceability, 0, 1)
            energy = np.clip(energy, 0, 1)
            valence = np.clip(valence, 0, 1)
            tempo = np.clip(tempo, 60, 200)
            
            # Generate other features
            loudness = np.random.normal(-10, 5)
            speechiness = np.random.normal(0.05, 0.02)
            acousticness = np.random.normal(0.3, 0.2)
            instrumentalness = np.random.normal(0.1, 0.1)
            liveness = np.random.normal(0.1, 0.05)
            popularity = np.random.randint(20, 100)
            
            data.append({
                'track_id': track_id,
                'track_name': f"{track} {i+1}",
                'artist_name': artist,
                'genre': genre,
                'danceability': danceability,
                'energy': energy,
                'valence': valence,
                'tempo': tempo,
                'loudness': loudness,
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'popularity': popularity
            })
            track_id += 1
    
    return pd.DataFrame(data)

# Load sample data
df = create_sample_data()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get recommendations based on user input"""
    try:
        data = request.get_json()
        track_name = data.get('track_name', '').strip()
        artist_name = data.get('artist_name', '').strip()
        top_n = int(data.get('top_n', 10))
        
        if not track_name:
            return jsonify({'error': 'Track name is required'}), 400
        
        # Find the track in our sample data
        mask = df['track_name'].str.contains(track_name, case=False, na=False)
        if artist_name:
            mask = mask & df['artist_name'].str.contains(artist_name, case=False, na=False)
        
        matching_tracks = df[mask]
        
        if len(matching_tracks) == 0:
            return jsonify({'error': 'Track not found in dataset'}), 404
        
        # Get the first matching track
        track_data = matching_tracks.iloc[0]
        
        # Get content-based recommendations using cosine similarity
        features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        # Get features for the selected track
        track_features = track_data[features].values.reshape(1, -1)
        
        # Get features for all other tracks
        other_tracks = df[df['track_name'] != track_data['track_name']]
        other_features = other_tracks[features].values
        
        # Calculate similarities
        similarities = cosine_similarity(track_features, other_features)[0]
        
        # Get top similar tracks
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            track = other_tracks.iloc[idx]
            recommendations.append({
                'track_name': track['track_name'],
                'artist_name': track['artist_name'],
                'genre': track['genre'],
                'similarity_score': similarities[idx],
                'popularity': track['popularity']
            })
        
        # Analyze recommendation quality
        analysis = {
            'total_recommendations': len(recommendations),
            'genre_distribution': {},
            'average_popularity': 0
        }
        
        # Genre distribution
        genres = [rec['genre'] for rec in recommendations]
        genre_counts = pd.Series(genres).value_counts()
        analysis['genre_distribution'] = genre_counts.to_dict()
        
        # Average popularity
        popularities = [rec['popularity'] for rec in recommendations]
        analysis['average_popularity'] = np.mean(popularities)
        
        return jsonify({
            'recommendations': recommendations,
            'analysis': analysis,
            'input_track': track_name,
            'input_artist': artist_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_statistics():
    """Get dataset statistics"""
    try:
        stats = {
            'total_tracks': len(df),
            'total_artists': df['artist_name'].nunique(),
            'total_genres': df['genre'].nunique(),
            'genre_distribution': df['genre'].value_counts().to_dict(),
            'average_popularity': df['popularity'].mean(),
            'popularity_range': {
                'min': df['popularity'].min(),
                'max': df['popularity'].max()
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search')
def search_tracks():
    """Search for tracks in the dataset"""
    try:
        query = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'tracks': []})
        
        # Search in track names and artist names
        mask = (df['track_name'].str.contains(query, case=False, na=False) |
                df['artist_name'].str.contains(query, case=False, na=False))
        
        results = df[mask].head(limit)
        
        tracks = []
        for _, track in results.iterrows():
            tracks.append({
                'track_name': track['track_name'],
                'artist_name': track['artist_name'],
                'genre': track['genre'],
                'popularity': track['popularity']
            })
        
        return jsonify({'tracks': tracks})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎵 Starting Spotify Recommendation System Demo...")
    print("✅ Sample data loaded successfully!")
    print("🌐 Flask app running on: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
