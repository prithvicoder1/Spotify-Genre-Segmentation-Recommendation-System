#!/usr/bin/env python3
"""
Demo of the Spotify Genre Segmentation Project without requiring Spotify API
This creates sample data to demonstrate the system functionality
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_data():
    """Create sample Spotify-like data for demonstration"""
    print("🎵 Creating sample music data for demonstration...")
    
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
            key = np.random.randint(0, 12)
            mode = np.random.randint(0, 2)
            time_signature = np.random.choice([3, 4, 5])
            popularity = np.random.randint(20, 100)
            
            data.append({
                'track_id': track_id,
                'track_name': f"{track} {i+1}",
                'artist_name': artist,
                'album_name': f"{artist} Album {i+1}",
                'genre': genre,
                'danceability': danceability,
                'energy': energy,
                'key': key,
                'loudness': loudness,
                'mode': mode,
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'valence': valence,
                'tempo': tempo,
                'time_signature': time_signature,
                'popularity': popularity
            })
            track_id += 1
    
    return pd.DataFrame(data)

def demonstrate_clustering(df):
    """Demonstrate clustering on the sample data"""
    print("\n🔬 Demonstrating Clustering Analysis...")
    
    # Select features for clustering
    features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    X = df[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Analyze clusters
    print("\n📊 Cluster Analysis:")
    cluster_stats = df.groupby('cluster')[features].mean()
    print(cluster_stats.round(3))
    
    # Show genre distribution in clusters
    print("\n🎭 Genre Distribution in Clusters:")
    genre_cluster = pd.crosstab(df['cluster'], df['genre'])
    print(genre_cluster)
    
    return df, kmeans, scaler

def demonstrate_recommendations(df, kmeans, scaler):
    """Demonstrate recommendation system"""
    print("\n🎯 Demonstrating Recommendation System...")
    
    # Select a sample track
    sample_track = df.iloc[0]
    print(f"\n🎵 Finding recommendations for: '{sample_track['track_name']}' by {sample_track['artist_name']}")
    print(f"Genre: {sample_track['genre']}")
    
    # Get features for the sample track
    features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    sample_features = sample_track[features].values.reshape(1, -1)
    sample_features_scaled = scaler.transform(sample_features)
    
    # Content-based recommendations (cosine similarity)
    X_scaled = scaler.transform(df[features].values)
    similarities = cosine_similarity(sample_features_scaled, X_scaled)[0]
    
    # Get top similar tracks (excluding the sample itself)
    similar_indices = np.argsort(similarities)[::-1][1:11]  # Top 10 excluding self
    
    print("\n📈 Content-Based Recommendations:")
    for i, idx in enumerate(similar_indices, 1):
        track = df.iloc[idx]
        print(f"{i}. {track['track_name']} - {track['artist_name']} ({track['genre']}) - Similarity: {similarities[idx]:.3f}")
    
    # Cluster-based recommendations
    sample_cluster = kmeans.predict(sample_features_scaled)[0]
    cluster_tracks = df[df['cluster'] == sample_cluster]
    cluster_tracks = cluster_tracks[cluster_tracks.index != sample_track.name]  # Exclude sample
    
    print(f"\n🎯 Cluster-Based Recommendations (Cluster {sample_cluster}):")
    for i, (_, track) in enumerate(cluster_tracks.head(5).iterrows(), 1):
        print(f"{i}. {track['track_name']} - {track['artist_name']} ({track['genre']})")
    
    return similar_indices, cluster_tracks

def create_visualizations(df):
    """Create sample visualizations"""
    print("\n📊 Creating Visualizations...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Genre distribution
    plt.figure(figsize=(12, 6))
    genre_counts = df['genre'].value_counts()
    plt.subplot(1, 2, 1)
    genre_counts.plot(kind='bar', color='skyblue')
    plt.title('Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Tracks')
    plt.xticks(rotation=45)
    
    # Audio features by genre
    plt.subplot(1, 2, 2)
    features = ['danceability', 'energy', 'valence']
    df_melted = df.melt(id_vars=['genre'], value_vars=features, var_name='feature', value_name='value')
    sns.boxplot(data=df_melted, x='genre', y='value', hue='feature')
    plt.title('Audio Features by Genre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('plots/demo_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
                     'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    corr_matrix = df[audio_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Audio Features Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/demo_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved to plots/ directory")

def main():
    """Main demonstration function"""
    print("🎵 Spotify Genre Segmentation - DEMO VERSION")
    print("=" * 60)
    print("This demo creates sample data to show how the system works")
    print("without requiring Spotify API credentials.")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"✅ Created sample dataset with {len(df)} tracks across {df['genre'].nunique()} genres")
    
    # Show basic statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Total tracks: {len(df)}")
    print(f"Total artists: {df['artist_name'].nunique()}")
    print(f"Total genres: {df['genre'].nunique()}")
    print(f"Average popularity: {df['popularity'].mean():.1f}")
    
    print(f"\n🎭 Genre distribution:")
    genre_counts = df['genre'].value_counts()
    for genre, count in genre_counts.items():
        print(f"  {genre}: {count} tracks")
    
    # Demonstrate clustering
    df, kmeans, scaler = demonstrate_clustering(df)
    
    # Demonstrate recommendations
    similar_indices, cluster_tracks = demonstrate_recommendations(df, kmeans, scaler)
    
    # Create visualizations
    create_visualizations(df)
    
    print(f"\n🎉 Demo completed successfully!")
    print("=" * 60)
    print("What you've seen:")
    print("✅ Sample data creation with realistic music features")
    print("✅ K-Means clustering for genre segmentation")
    print("✅ Content-based recommendations using cosine similarity")
    print("✅ Cluster-based recommendations")
    print("✅ Data visualizations and analysis")
    print("\nTo use the full system with real Spotify data:")
    print("1. Get Spotify API credentials from https://developer.spotify.com/dashboard")
    print("2. Update config.py with your credentials")
    print("3. Run: python spotify_recommendation_project.py")
    
    return df

if __name__ == "__main__":
    df = main()
