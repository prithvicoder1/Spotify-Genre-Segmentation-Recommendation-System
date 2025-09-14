from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from recommendation_system import SpotifyRecommendationSystem
from config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize recommendation system
rec_system = None

def initialize_recommendation_system():
    """Initialize the recommendation system"""
    global rec_system
    try:
        rec_system = SpotifyRecommendationSystem()
        return True
    except Exception as e:
        print(f"Error initializing recommendation system: {str(e)}")
        return False

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
        recommendation_type = data.get('type', 'hybrid')
        top_n = int(data.get('top_n', 10))
        
        if not track_name:
            return jsonify({'error': 'Track name is required'}), 400
        
        if not rec_system:
            return jsonify({'error': 'Recommendation system not initialized'}), 500
        
        # Get recommendations based on type
        if recommendation_type == 'content':
            recommendations = rec_system.get_content_recommendations(
                track_name, artist_name, top_n
            )
        elif recommendation_type == 'cluster':
            recommendations = rec_system.get_cluster_recommendations(
                track_name, artist_name, top_n
            )
        elif recommendation_type == 'diverse':
            recommendations = rec_system.get_diverse_recommendations(
                track_name, artist_name, top_n
            )
        else:  # hybrid
            recommendations = rec_system.get_hybrid_recommendations(
                track_name, artist_name, top_n=top_n
            )
        
        # Analyze recommendation quality
        analysis = rec_system.analyze_recommendation_quality(recommendations)
        
        return jsonify({
            'recommendations': recommendations,
            'analysis': analysis,
            'input_track': track_name,
            'input_artist': artist_name,
            'type': recommendation_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/genre/<genre_name>')
def get_genre_recommendations(genre_name):
    """Get recommendations for a specific genre"""
    try:
        if not rec_system:
            return jsonify({'error': 'Recommendation system not initialized'}), 500
        
        recommendations = rec_system.get_genre_recommendations(genre_name, top_n=20)
        
        return jsonify({
            'genre': genre_name,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/artist/<artist_name>')
def get_artist_recommendations(artist_name):
    """Get recommendations based on an artist"""
    try:
        if not rec_system:
            return jsonify({'error': 'Recommendation system not initialized'}), 500
        
        recommendations = rec_system.get_artist_recommendations(artist_name, top_n=20)
        
        return jsonify({
            'artist': artist_name,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/genres')
def get_available_genres():
    """Get list of available genres"""
    try:
        if not rec_system or rec_system.data is None:
            return jsonify({'error': 'Data not available'}), 500
        
        genres = rec_system.data['genre'].unique().tolist()
        genre_counts = rec_system.data['genre'].value_counts().to_dict()
        
        return jsonify({
            'genres': genres,
            'genre_counts': genre_counts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_statistics():
    """Get dataset statistics"""
    try:
        if not rec_system or rec_system.data is None:
            return jsonify({'error': 'Data not available'}), 500
        
        stats = {
            'total_tracks': len(rec_system.data),
            'total_artists': rec_system.data['artist_name'].nunique(),
            'total_genres': rec_system.data['genre'].nunique(),
            'genre_distribution': rec_system.data['genre'].value_counts().to_dict(),
            'average_popularity': rec_system.data['popularity'].mean(),
            'popularity_range': {
                'min': rec_system.data['popularity'].min(),
                'max': rec_system.data['popularity'].max()
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
        
        if not query or not rec_system or rec_system.data is None:
            return jsonify({'tracks': []})
        
        # Search in track names and artist names
        mask = (rec_system.data['track_name'].str.contains(query, case=False, na=False) |
                rec_system.data['artist_name'].str.contains(query, case=False, na=False))
        
        results = rec_system.data[mask].head(limit)
        
        tracks = []
        for _, track in results.iterrows():
            tracks.append({
                'track_name': track['track_name'],
                'artist_name': track['artist_name'],
                'genre': track['genre'],
                'popularity': track.get('popularity', 0)
            })
        
        return jsonify({'tracks': tracks})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Initializing Spotify Recommendation System...")
    
    # Initialize recommendation system
    if initialize_recommendation_system():
        print("Recommendation system initialized successfully!")
        print("Starting Flask application...")
        print("Visit: http://localhost:5000")
        app.run(debug=Config.FLASK_DEBUG, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize recommendation system!")
        print("Please ensure you have:")
        print("1. Run data_collector.py to collect data")
        print("2. Run data_preprocessing.py to preprocess data")
        print("3. Run clustering_models.py to train models")
        print("4. Set up your Spotify API credentials in config.py")
