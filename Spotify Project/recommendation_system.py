import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
import os
from config import Config

class SpotifyRecommendationSystem:
    def __init__(self):
        """Initialize the recommendation system"""
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.cluster_labels = None
        
        # Spotify API setup
        self.client_credentials_manager = SpotifyClientCredentials(
            client_id=Config.SPOTIPY_CLIENT_ID,
            client_secret=Config.SPOTIPY_CLIENT_SECRET
        )
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        
        # Load models and data
        self.load_models_and_data()
    
    def load_models_and_data(self):
        """Load trained models and processed data"""
        print("Loading models and data...")
        
        # Load processed data
        data_file = os.path.join(Config.DATA_DIR, 'processed_full_data.csv')
        features_file = os.path.join(Config.DATA_DIR, 'processed_features.csv')
        
        if os.path.exists(data_file):
            self.data = pd.read_csv(data_file)
            print(f"Loaded data with {len(self.data)} tracks")
        
        if os.path.exists(features_file):
            self.features = pd.read_csv(features_file)
            print(f"Loaded features with shape: {self.features.shape}")
        
        # Load best clustering model (K-Means by default)
        model_file = os.path.join(Config.MODELS_DIR, 'kmeans_model.pkl')
        labels_file = os.path.join(Config.MODELS_DIR, 'kmeans_labels.pkl')
        
        if os.path.exists(model_file) and os.path.exists(labels_file):
            self.clustering_model = joblib.load(model_file)
            self.cluster_labels = joblib.load(labels_file)
            print("Loaded clustering model and labels")
        else:
            print("Clustering model not found. Please run clustering_models.py first.")
    
    def get_track_features(self, track_name, artist_name=None):
        """Get audio features for a specific track"""
        try:
            # Search for the track
            if artist_name:
                query = f"track:{track_name} artist:{artist_name}"
            else:
                query = f"track:{track_name}"
            
            results = self.sp.search(q=query, type='track', limit=1)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_id = track['id']
                
                # Get audio features
                features = self.sp.audio_features(track_id)[0]
                
                if features:
                    return {
                        'track_id': track_id,
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'danceability': features['danceability'],
                        'energy': features['energy'],
                        'key': features['key'],
                        'loudness': features['loudness'],
                        'mode': features['mode'],
                        'speechiness': features['speechiness'],
                        'acousticness': features['acousticness'],
                        'instrumentalness': features['instrumentalness'],
                        'liveness': features['liveness'],
                        'valence': features['valence'],
                        'tempo': features['tempo'],
                        'time_signature': features['time_signature']
                    }
        except Exception as e:
            print(f"Error getting track features: {str(e)}")
        
        return None
    
    def find_track_in_dataset(self, track_name, artist_name=None):
        """Find a track in the existing dataset"""
        if self.data is None:
            return None
        
        # Try exact match first
        if artist_name:
            mask = (self.data['track_name'].str.lower() == track_name.lower()) & \
                   (self.data['artist_name'].str.lower() == artist_name.lower())
        else:
            mask = self.data['track_name'].str.lower() == track_name.lower()
        
        matches = self.data[mask]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Try partial match
        if artist_name:
            mask = (self.data['track_name'].str.contains(track_name, case=False, na=False)) & \
                   (self.data['artist_name'].str.contains(artist_name, case=False, na=False))
        else:
            mask = self.data['track_name'].str.contains(track_name, case=False, na=False)
        
        matches = self.data[mask]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        return None
    
    def get_content_recommendations(self, track_name, artist_name=None, top_n=10):
        """Get content-based recommendations using audio features similarity"""
        print(f"Getting content-based recommendations for: {track_name}")
        
        # Find track in dataset or get features from Spotify
        track_data = self.find_track_in_dataset(track_name, artist_name)
        
        if track_data is None:
            # Get features from Spotify API
            track_features = self.get_track_features(track_name, artist_name)
            if track_features is None:
                print("Track not found in dataset or Spotify API")
                return []
            
            # Convert to DataFrame format
            track_data = pd.Series(track_features)
        
        # Get audio features for similarity calculation
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Filter existing features
        audio_features = [col for col in audio_features if col in track_data.index]
        
        # Get track features vector
        track_vector = track_data[audio_features].values.reshape(1, -1)
        
        # Get all other tracks' features
        if self.data is not None:
            other_tracks = self.data[self.data['track_name'] != track_name]
            other_features = other_tracks[audio_features].values
            
            # Calculate cosine similarity
            similarities = cosine_similarity(track_vector, other_features)[0]
            
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
                    'popularity': track.get('popularity', 0)
                })
            
            return recommendations
        
        return []
    
    def get_cluster_recommendations(self, track_name, artist_name=None, top_n=10):
        """Get cluster-based recommendations"""
        print(f"Getting cluster-based recommendations for: {track_name}")
        
        if self.clustering_model is None or self.cluster_labels is None:
            print("Clustering model not available")
            return []
        
        # Find track in dataset
        track_data = self.find_track_in_dataset(track_name, artist_name)
        
        if track_data is None:
            print("Track not found in dataset")
            return []
        
        # Get track index
        track_idx = track_data.name
        
        # Get cluster of the input track
        track_cluster = self.cluster_labels[track_idx]
        
        # Find other tracks in the same cluster
        same_cluster_indices = np.where(self.cluster_labels == track_cluster)[0]
        
        # Remove the input track itself
        same_cluster_indices = same_cluster_indices[same_cluster_indices != track_idx]
        
        if len(same_cluster_indices) == 0:
            print("No other tracks found in the same cluster")
            return []
        
        # Get recommendations from the same cluster
        recommendations = []
        for idx in same_cluster_indices[:top_n]:
            track = self.data.iloc[idx]
            recommendations.append({
                'track_name': track['track_name'],
                'artist_name': track['artist_name'],
                'genre': track['genre'],
                'cluster': track_cluster,
                'popularity': track.get('popularity', 0)
            })
        
        return recommendations
    
    def get_hybrid_recommendations(self, track_name, artist_name=None, 
                                 content_weight=Config.CONTENT_WEIGHT, 
                                 cluster_weight=Config.CLUSTER_WEIGHT, 
                                 top_n=Config.TOP_N_RECOMMENDATIONS):
        """Get hybrid recommendations combining content-based and cluster-based approaches"""
        print(f"Getting hybrid recommendations for: {track_name}")
        
        # Get content-based recommendations
        content_recs = self.get_content_recommendations(track_name, artist_name, top_n=top_n*2)
        
        # Get cluster-based recommendations
        cluster_recs = self.get_cluster_recommendations(track_name, artist_name, top_n=top_n*2)
        
        # Combine recommendations with weighted scoring
        hybrid_scores = {}
        
        # Process content-based recommendations
        for rec in content_recs:
            key = f"{rec['track_name']}_{rec['artist_name']}"
            hybrid_scores[key] = {
                'track_name': rec['track_name'],
                'artist_name': rec['artist_name'],
                'genre': rec['genre'],
                'content_score': rec['similarity_score'],
                'cluster_score': 0,
                'popularity': rec['popularity']
            }
        
        # Process cluster-based recommendations
        for rec in cluster_recs:
            key = f"{rec['track_name']}_{rec['artist_name']}"
            if key in hybrid_scores:
                hybrid_scores[key]['cluster_score'] = 1.0  # Same cluster
            else:
                hybrid_scores[key] = {
                    'track_name': rec['track_name'],
                    'artist_name': rec['artist_name'],
                    'genre': rec['genre'],
                    'content_score': 0,
                    'cluster_score': 1.0,
                    'popularity': rec['popularity']
                }
        
        # Calculate hybrid scores
        for key, rec in hybrid_scores.items():
            hybrid_score = (content_weight * rec['content_score'] + 
                          cluster_weight * rec['cluster_score'])
            rec['hybrid_score'] = hybrid_score
        
        # Sort by hybrid score and return top recommendations
        sorted_recommendations = sorted(hybrid_scores.values(), 
                                      key=lambda x: x['hybrid_score'], 
                                      reverse=True)
        
        return sorted_recommendations[:top_n]
    
    def get_genre_recommendations(self, genre, top_n=10):
        """Get recommendations based on genre"""
        print(f"Getting genre-based recommendations for: {genre}")
        
        if self.data is None:
            return []
        
        # Filter tracks by genre
        genre_tracks = self.data[self.data['genre'] == genre]
        
        if len(genre_tracks) == 0:
            print(f"No tracks found for genre: {genre}")
            return []
        
        # Sort by popularity and return top tracks
        top_tracks = genre_tracks.nlargest(top_n, 'popularity')
        
        recommendations = []
        for _, track in top_tracks.iterrows():
            recommendations.append({
                'track_name': track['track_name'],
                'artist_name': track['artist_name'],
                'genre': track['genre'],
                'popularity': track.get('popularity', 0)
            })
        
        return recommendations
    
    def get_artist_recommendations(self, artist_name, top_n=10):
        """Get recommendations based on artist"""
        print(f"Getting artist-based recommendations for: {artist_name}")
        
        if self.data is None:
            return []
        
        # Find tracks by the artist
        artist_tracks = self.data[self.data['artist_name'].str.contains(artist_name, case=False, na=False)]
        
        if len(artist_tracks) == 0:
            print(f"No tracks found for artist: {artist_name}")
            return []
        
        # Get the most popular track by the artist
        top_track = artist_tracks.loc[artist_tracks['popularity'].idxmax()]
        
        # Get hybrid recommendations based on the top track
        recommendations = self.get_hybrid_recommendations(
            top_track['track_name'], 
            top_track['artist_name'], 
            top_n=top_n
        )
        
        return recommendations
    
    def analyze_recommendation_quality(self, recommendations, target_genre=None):
        """Analyze the quality of recommendations"""
        if not recommendations:
            return {}
        
        analysis = {
            'total_recommendations': len(recommendations),
            'genre_distribution': {},
            'average_popularity': 0,
            'genre_accuracy': 0
        }
        
        # Genre distribution
        genres = [rec['genre'] for rec in recommendations]
        genre_counts = pd.Series(genres).value_counts()
        analysis['genre_distribution'] = genre_counts.to_dict()
        
        # Average popularity
        popularities = [rec.get('popularity', 0) for rec in recommendations]
        analysis['average_popularity'] = np.mean(popularities)
        
        # Genre accuracy (if target genre specified)
        if target_genre:
            matching_genre = sum(1 for rec in recommendations if rec['genre'] == target_genre)
            analysis['genre_accuracy'] = matching_genre / len(recommendations)
        
        return analysis
    
    def get_diverse_recommendations(self, track_name, artist_name=None, top_n=10):
        """Get diverse recommendations across different genres"""
        print(f"Getting diverse recommendations for: {track_name}")
        
        # Get hybrid recommendations
        all_recs = self.get_hybrid_recommendations(track_name, artist_name, top_n=top_n*3)
        
        # Group by genre
        genre_groups = {}
        for rec in all_recs:
            genre = rec['genre']
            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append(rec)
        
        # Select top recommendations from each genre
        diverse_recs = []
        genres = list(genre_groups.keys())
        
        # Round-robin selection to ensure diversity
        max_per_genre = max(1, top_n // len(genres))
        
        for genre in genres:
            genre_recs = sorted(genre_groups[genre], 
                              key=lambda x: x['hybrid_score'], 
                              reverse=True)[:max_per_genre]
            diverse_recs.extend(genre_recs)
        
        # Sort by hybrid score and return top N
        diverse_recs = sorted(diverse_recs, 
                            key=lambda x: x['hybrid_score'], 
                            reverse=True)[:top_n]
        
        return diverse_recs

def main():
    """Main function to test the recommendation system"""
    print("Initializing Spotify Recommendation System...")
    
    # Initialize recommendation system
    rec_system = SpotifyRecommendationSystem()
    
    # Test with a sample track
    test_track = "Shape of You"
    test_artist = "Ed Sheeran"
    
    print(f"\nTesting recommendations for: {test_track} by {test_artist}")
    
    # Get different types of recommendations
    print("\n1. Content-based recommendations:")
    content_recs = rec_system.get_content_recommendations(test_track, test_artist)
    for i, rec in enumerate(content_recs[:5], 1):
        print(f"{i}. {rec['track_name']} - {rec['artist_name']} ({rec['genre']}) - Score: {rec['similarity_score']:.3f}")
    
    print("\n2. Cluster-based recommendations:")
    cluster_recs = rec_system.get_cluster_recommendations(test_track, test_artist)
    for i, rec in enumerate(cluster_recs[:5], 1):
        print(f"{i}. {rec['track_name']} - {rec['artist_name']} ({rec['genre']}) - Cluster: {rec['cluster']}")
    
    print("\n3. Hybrid recommendations:")
    hybrid_recs = rec_system.get_hybrid_recommendations(test_track, test_artist)
    for i, rec in enumerate(hybrid_recs[:5], 1):
        print(f"{i}. {rec['track_name']} - {rec['artist_name']} ({rec['genre']}) - Score: {rec['hybrid_score']:.3f}")
    
    print("\n4. Diverse recommendations:")
    diverse_recs = rec_system.get_diverse_recommendations(test_track, test_artist)
    for i, rec in enumerate(diverse_recs[:5], 1):
        print(f"{i}. {rec['track_name']} - {rec['artist_name']} ({rec['genre']}) - Score: {rec['hybrid_score']:.3f}")
    
    # Analyze recommendation quality
    print("\n5. Recommendation quality analysis:")
    analysis = rec_system.analyze_recommendation_quality(hybrid_recs)
    print(f"Total recommendations: {analysis['total_recommendations']}")
    print(f"Average popularity: {analysis['average_popularity']:.1f}")
    print(f"Genre distribution: {analysis['genre_distribution']}")
    
    return rec_system

if __name__ == "__main__":
    rec_system = main()
