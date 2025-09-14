import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import time
import os
from config import Config

class SpotifyDataCollector:
    def __init__(self):
        """Initialize Spotify API client"""
        self.client_credentials_manager = SpotifyClientCredentials(
            client_id=Config.SPOTIPY_CLIENT_ID,
            client_secret=Config.SPOTIPY_CLIENT_SECRET
        )
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        
    def get_playlist_tracks(self, playlist_id):
        """Extract tracks from a Spotify playlist"""
        tracks = []
        results = self.sp.playlist_tracks(playlist_id)
        
        while results:
            for item in results['items']:
                if item['track']:
                    track = item['track']
                    tracks.append({
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'album_name': track['album']['name'],
                        'release_date': track['album']['release_date'],
                        'popularity': track['popularity'],
                        'duration_ms': track['duration_ms']
                    })
            
            if results['next']:
                results = self.sp.next(results)
            else:
                break
                
        return tracks
    
    def get_audio_features(self, track_ids):
        """Get audio features for a list of track IDs"""
        features = []
        
        # Process in batches of 100 (Spotify API limit)
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            batch_features = self.sp.audio_features(batch)
            
            for feature in batch_features:
                if feature:
                    features.append({
                        'track_id': feature['id'],
                        'danceability': feature['danceability'],
                        'energy': feature['energy'],
                        'key': feature['key'],
                        'loudness': feature['loudness'],
                        'mode': feature['mode'],
                        'speechiness': feature['speechiness'],
                        'acousticness': feature['acousticness'],
                        'instrumentalness': feature['instrumentalness'],
                        'liveness': feature['liveness'],
                        'valence': feature['valence'],
                        'tempo': feature['tempo'],
                        'time_signature': feature['time_signature']
                    })
            
            # Rate limiting
            time.sleep(0.1)
            
        return features
    
    def collect_genre_data(self, genre_playlists):
        """Collect data for multiple genres"""
        all_data = []
        
        for genre, playlist_id in genre_playlists.items():
            print(f"Collecting data for {genre}...")
            
            # Get tracks from playlist
            tracks = self.get_playlist_tracks(playlist_id)
            track_ids = [track['track_id'] for track in tracks]
            
            # Get audio features
            features = self.get_audio_features(track_ids)
            
            # Combine track info with audio features
            for track in tracks:
                track_features = next((f for f in features if f['track_id'] == track['track_id']), None)
                if track_features:
                    combined_data = {**track, **track_features}
                    combined_data['genre'] = genre
                    all_data.append(combined_data)
            
            print(f"Collected {len(tracks)} tracks for {genre}")
            time.sleep(1)  # Rate limiting
        
        return pd.DataFrame(all_data)
    
    def save_data(self, df, filename='spotify_data.csv'):
        """Save collected data to CSV"""
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        filepath = os.path.join(Config.DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

# Sample genre playlists (you can replace these with your own)
SAMPLE_PLAYLISTS = {
    'pop': '37i9dQZF1DXcBWIGoYBM5M',  # Today's Top Hits
    'rock': '37i9dQZF1DXcF6B6QPhFDv',  # Rock Classics
    'hip_hop': '37i9dQZF1DX0XUsuxWHRQd',  # RapCaviar
    'electronic': '37i9dQZF1DX4dyzvuaRJ0n',  # mint
    'jazz': '37i9dQZF1DXbITWG1ZJKYt',  # Jazz Classics
    'classical': '37i9dQZF1DX7K31D69s4M1',  # Classical Essentials
    'country': '37i9dQZF1DX1lVhptI8daE',  # Hot Country
    'r&b': '37i9dQZF1DX4SBhb3fqCJd'  # Are & Be
}

def main():
    """Main function to collect Spotify data"""
    collector = SpotifyDataCollector()
    
    print("Starting Spotify data collection...")
    df = collector.collect_genre_data(SAMPLE_PLAYLISTS)
    
    print(f"Collected {len(df)} tracks total")
    print(f"Genres: {df['genre'].value_counts().to_dict()}")
    
    # Save data
    filepath = collector.save_data(df)
    
    return df

if __name__ == "__main__":
    df = main()
