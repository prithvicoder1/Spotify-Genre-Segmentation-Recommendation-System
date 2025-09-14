#!/usr/bin/env python3
"""
Example usage of the Spotify Genre Segmentation & Recommendation System
"""

import os
import sys
from recommendation_system import SpotifyRecommendationSystem

def main():
    """Demonstrate the recommendation system"""
    print("🎵 Spotify Recommendation System - Example Usage")
    print("=" * 60)
    
    # Initialize the recommendation system
    print("🔄 Initializing recommendation system...")
    try:
        rec_system = SpotifyRecommendationSystem()
        print("✅ Recommendation system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing recommendation system: {e}")
        print("Please ensure you have:")
        print("1. Run the complete pipeline: python spotify_recommendation_project.py")
        print("2. Or run individual components in order")
        return
    
    # Example tracks to test
    example_tracks = [
        ("Shape of You", "Ed Sheeran"),
        ("Blinding Lights", "The Weeknd"),
        ("Watermelon Sugar", "Harry Styles"),
        ("Levitating", "Dua Lipa"),
        ("Good 4 U", "Olivia Rodrigo")
    ]
    
    print(f"\n🧪 Testing with {len(example_tracks)} example tracks...")
    print("-" * 60)
    
    for i, (track_name, artist_name) in enumerate(example_tracks, 1):
        print(f"\n{i}. Testing: '{track_name}' by {artist_name}")
        print("-" * 40)
        
        try:
            # Get hybrid recommendations
            recommendations = rec_system.get_hybrid_recommendations(
                track_name, artist_name, top_n=5
            )
            
            if recommendations:
                print(f"✅ Found {len(recommendations)} recommendations:")
                for j, rec in enumerate(recommendations, 1):
                    score = rec.get('hybrid_score', 0)
                    print(f"   {j}. {rec['track_name']} - {rec['artist_name']}")
                    print(f"      Genre: {rec['genre']}, Score: {score:.3f}")
            else:
                print("⚠️ No recommendations found")
                
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
    
    # Demonstrate different recommendation types
    print(f"\n🎯 Demonstrating different recommendation types...")
    print("-" * 60)
    
    test_track = "Shape of You"
    test_artist = "Ed Sheeran"
    
    recommendation_types = [
        ("Hybrid", "hybrid"),
        ("Content-Based", "content"),
        ("Cluster-Based", "cluster"),
        ("Diverse", "diverse")
    ]
    
    for type_name, type_key in recommendation_types:
        print(f"\n📊 {type_name} Recommendations for '{test_track}':")
        print("-" * 30)
        
        try:
            if type_key == "content":
                recommendations = rec_system.get_content_recommendations(test_track, test_artist, top_n=3)
            elif type_key == "cluster":
                recommendations = rec_system.get_cluster_recommendations(test_track, test_artist, top_n=3)
            elif type_key == "diverse":
                recommendations = rec_system.get_diverse_recommendations(test_track, test_artist, top_n=3)
            else:  # hybrid
                recommendations = rec_system.get_hybrid_recommendations(test_track, test_artist, top_n=3)
            
            if recommendations:
                for j, rec in enumerate(recommendations, 1):
                    score = rec.get('hybrid_score', rec.get('similarity_score', rec.get('cluster_score', 0)))
                    print(f"   {j}. {rec['track_name']} - {rec['artist_name']} ({rec['genre']}) - Score: {score:.3f}")
            else:
                print("   No recommendations found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Demonstrate genre-based recommendations
    print(f"\n🎭 Genre-based Recommendations:")
    print("-" * 40)
    
    if rec_system.data is not None:
        genres = rec_system.data['genre'].unique()[:3]  # Test first 3 genres
        
        for genre in genres:
            print(f"\n📀 {genre.title()} Recommendations:")
            try:
                recommendations = rec_system.get_genre_recommendations(genre, top_n=3)
                if recommendations:
                    for j, rec in enumerate(recommendations, 1):
                        print(f"   {j}. {rec['track_name']} - {rec['artist_name']} (Popularity: {rec['popularity']})")
                else:
                    print("   No recommendations found")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Show dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print("-" * 30)
    
    if rec_system.data is not None:
        print(f"Total tracks: {len(rec_system.data):,}")
        print(f"Total artists: {rec_system.data['artist_name'].nunique():,}")
        print(f"Total genres: {rec_system.data['genre'].nunique()}")
        print(f"Average popularity: {rec_system.data['popularity'].mean():.1f}")
        
        print(f"\nGenre distribution:")
        genre_counts = rec_system.data['genre'].value_counts()
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} tracks")
    else:
        print("No data available")
    
    print(f"\n🎉 Example usage completed!")
    print("=" * 60)
    print("Next steps:")
    print("1. 🌐 Run Flask app: python flask_app.py")
    print("2. 📱 Run Streamlit app: streamlit run streamlit_app.py")
    print("3. 🔍 Explore the generated visualizations in the plots/ directory")

if __name__ == "__main__":
    main()
