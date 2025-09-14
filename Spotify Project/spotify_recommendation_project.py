#!/usr/bin/env python3
"""
Spotify Songs' Genre Segmentation Project
Complete ML Pipeline for Music Recommendation System

This script runs the complete pipeline:
1. Data Collection
2. Data Preprocessing  
3. Exploratory Data Analysis
4. Clustering Analysis
5. Recommendation System Training

Author: Your Name
Date: 2024
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Import our custom modules
from config import Config
from data_collector import SpotifyDataCollector, SAMPLE_PLAYLISTS
from data_preprocessing import DataPreprocessor
from eda_visualization import EDAAnalyzer
from clustering_models import ClusteringAnalyzer
from recommendation_system import SpotifyRecommendationSystem

class SpotifyProjectPipeline:
    def __init__(self):
        """Initialize the complete project pipeline"""
        self.start_time = time.time()
        self.results = {}
        
        # Create necessary directories
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        
        print("🎵 Spotify Genre Segmentation Project Pipeline")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def step_1_data_collection(self):
        """Step 1: Collect data from Spotify API"""
        print("\n📊 STEP 1: DATA COLLECTION")
        print("-" * 40)
        
        try:
            collector = SpotifyDataCollector()
            
            # Check if data already exists
            data_file = os.path.join(Config.DATA_DIR, 'spotify_data.csv')
            if os.path.exists(data_file):
                print("✅ Data file already exists. Skipping data collection.")
                print(f"📁 Data file: {data_file}")
                return True
            
            print("🔄 Collecting data from Spotify API...")
            print(f"📋 Genres to collect: {list(SAMPLE_PLAYLISTS.keys())}")
            
            df = collector.collect_genre_data(SAMPLE_PLAYLISTS)
            
            if len(df) == 0:
                print("❌ No data collected. Please check your Spotify API credentials.")
                return False
            
            # Save data
            filepath = collector.save_data(df)
            
            self.results['data_collection'] = {
                'status': 'success',
                'total_tracks': len(df),
                'genres': df['genre'].value_counts().to_dict(),
                'filepath': filepath
            }
            
            print(f"✅ Data collection completed successfully!")
            print(f"📊 Total tracks collected: {len(df)}")
            print(f"🎭 Genres: {df['genre'].value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in data collection: {str(e)}")
            self.results['data_collection'] = {'status': 'error', 'error': str(e)}
            return False
    
    def step_2_data_preprocessing(self):
        """Step 2: Preprocess the collected data"""
        print("\n🔧 STEP 2: DATA PREPROCESSING")
        print("-" * 40)
        
        try:
            preprocessor = DataPreprocessor()
            
            # Check if processed data already exists
            processed_file = os.path.join(Config.DATA_DIR, 'processed_features.csv')
            if os.path.exists(processed_file):
                print("✅ Processed data already exists. Skipping preprocessing.")
                return True
            
            # Check if raw data exists
            data_file = os.path.join(Config.DATA_DIR, 'spotify_data.csv')
            if not os.path.exists(data_file):
                print("❌ Raw data file not found. Please run data collection first.")
                return False
            
            print("🔄 Preprocessing data...")
            
            # Run preprocessing pipeline
            df, X_scaled, y = preprocessor.preprocess_pipeline(data_file)
            
            # Save processed data
            preprocessor.save_processed_data(df, X_scaled, y)
            
            self.results['data_preprocessing'] = {
                'status': 'success',
                'original_shape': df.shape,
                'features_shape': X_scaled.shape,
                'target_shape': y.shape if y is not None else None
            }
            
            print(f"✅ Data preprocessing completed successfully!")
            print(f"📊 Original data shape: {df.shape}")
            print(f"🔢 Features shape: {X_scaled.shape}")
            print(f"🎯 Target shape: {y.shape if y is not None else 'None'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in data preprocessing: {str(e)}")
            self.results['data_preprocessing'] = {'status': 'error', 'error': str(e)}
            return False
    
    def step_3_exploratory_data_analysis(self):
        """Step 3: Perform exploratory data analysis"""
        print("\n📈 STEP 3: EXPLORATORY DATA ANALYSIS")
        print("-" * 40)
        
        try:
            analyzer = EDAAnalyzer()
            
            # Check if processed data exists
            processed_file = os.path.join(Config.DATA_DIR, 'processed_full_data.csv')
            if not os.path.exists(processed_file):
                print("❌ Processed data file not found. Please run preprocessing first.")
                return False
            
            print("🔄 Performing exploratory data analysis...")
            
            # Run complete EDA
            df, corr_matrix, genre_stats, summary_report = analyzer.run_complete_eda(processed_file)
            
            self.results['eda'] = {
                'status': 'success',
                'correlation_matrix': corr_matrix,
                'genre_stats': genre_stats,
                'summary_report': summary_report
            }
            
            print(f"✅ Exploratory data analysis completed successfully!")
            print(f"📊 Generated visualizations in: {Config.PLOTS_DIR}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in EDA: {str(e)}")
            self.results['eda'] = {'status': 'error', 'error': str(e)}
            return False
    
    def step_4_clustering_analysis(self):
        """Step 4: Perform clustering analysis"""
        print("\n🔬 STEP 4: CLUSTERING ANALYSIS")
        print("-" * 40)
        
        try:
            clustering_analyzer = ClusteringAnalyzer()
            
            # Check if processed data exists
            features_file = os.path.join(Config.DATA_DIR, 'processed_features.csv')
            target_file = os.path.join(Config.DATA_DIR, 'processed_target.csv')
            
            if not os.path.exists(features_file):
                print("❌ Processed features file not found. Please run preprocessing first.")
                return False
            
            print("🔄 Performing clustering analysis...")
            
            # Run complete clustering analysis
            results = clustering_analyzer.run_complete_clustering_analysis(features_file, target_file)
            
            self.results['clustering'] = {
                'status': 'success',
                'results': results
            }
            
            print(f"✅ Clustering analysis completed successfully!")
            print(f"🤖 Trained models saved in: {Config.MODELS_DIR}")
            print(f"📊 Clustering visualizations in: {Config.PLOTS_DIR}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in clustering analysis: {str(e)}")
            self.results['clustering'] = {'status': 'error', 'error': str(e)}
            return False
    
    def step_5_recommendation_system(self):
        """Step 5: Test the recommendation system"""
        print("\n🎯 STEP 5: RECOMMENDATION SYSTEM")
        print("-" * 40)
        
        try:
            print("🔄 Initializing recommendation system...")
            
            # Initialize recommendation system
            rec_system = SpotifyRecommendationSystem()
            
            # Test with sample tracks
            test_tracks = [
                ("Shape of You", "Ed Sheeran"),
                ("Blinding Lights", "The Weeknd"),
                ("Watermelon Sugar", "Harry Styles")
            ]
            
            print("🧪 Testing recommendation system with sample tracks...")
            
            test_results = {}
            for track_name, artist_name in test_tracks:
                print(f"  🎵 Testing: {track_name} by {artist_name}")
                
                # Get hybrid recommendations
                recommendations = rec_system.get_hybrid_recommendations(track_name, artist_name, top_n=5)
                
                if recommendations:
                    test_results[f"{track_name}_{artist_name}"] = {
                        'recommendations_count': len(recommendations),
                        'top_recommendation': recommendations[0]['track_name'] if recommendations else None
                    }
                    print(f"    ✅ Found {len(recommendations)} recommendations")
                else:
                    print(f"    ⚠️ No recommendations found")
            
            self.results['recommendation_system'] = {
                'status': 'success',
                'test_results': test_results
            }
            
            print(f"✅ Recommendation system testing completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in recommendation system: {str(e)}")
            self.results['recommendation_system'] = {'status': 'error', 'error': str(e)}
            return False
    
    def generate_final_report(self):
        """Generate final project report"""
        print("\n📋 FINAL PROJECT REPORT")
        print("=" * 60)
        
        total_time = time.time() - self.start_time
        
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Summary of each step
        steps = [
            ('Data Collection', 'data_collection'),
            ('Data Preprocessing', 'data_preprocessing'),
            ('Exploratory Data Analysis', 'eda'),
            ('Clustering Analysis', 'clustering'),
            ('Recommendation System', 'recommendation_system')
        ]
        
        successful_steps = 0
        for step_name, step_key in steps:
            if step_key in self.results:
                status = self.results[step_key]['status']
                if status == 'success':
                    print(f"✅ {step_name}: SUCCESS")
                    successful_steps += 1
                else:
                    print(f"❌ {step_name}: FAILED - {self.results[step_key].get('error', 'Unknown error')}")
            else:
                print(f"⏭️ {step_name}: SKIPPED")
        
        print()
        print(f"📊 Overall Success Rate: {successful_steps}/{len(steps)} ({successful_steps/len(steps)*100:.1f}%)")
        
        # Project outputs
        print("\n📁 PROJECT OUTPUTS:")
        print(f"  📊 Data files: {Config.DATA_DIR}/")
        print(f"  🤖 Trained models: {Config.MODELS_DIR}/")
        print(f"  📈 Visualizations: {Config.PLOTS_DIR}/")
        
        # Next steps
        print("\n🚀 NEXT STEPS:")
        if successful_steps == len(steps):
            print("  🎉 All steps completed successfully!")
            print("  🌐 Run Flask app: python flask_app.py")
            print("  📱 Run Streamlit app: streamlit run streamlit_app.py")
            print("  🔍 Explore the generated visualizations and models")
        else:
            print("  🔧 Fix any failed steps before proceeding")
            print("  📖 Check the error messages above for guidance")
        
        print("\n" + "=" * 60)
        print("🎵 Spotify Genre Segmentation Project Complete!")
        print("=" * 60)
    
    def run_complete_pipeline(self):
        """Run the complete project pipeline"""
        print("🚀 Starting complete Spotify Genre Segmentation pipeline...")
        
        # Run all steps
        steps = [
            self.step_1_data_collection,
            self.step_2_data_preprocessing,
            self.step_3_exploratory_data_analysis,
            self.step_4_clustering_analysis,
            self.step_5_recommendation_system
        ]
        
        for step in steps:
            try:
                success = step()
                if not success:
                    print(f"⚠️ Step failed, but continuing with remaining steps...")
            except Exception as e:
                print(f"❌ Unexpected error in step: {str(e)}")
                continue
        
        # Generate final report
        self.generate_final_report()
        
        return self.results

def main():
    """Main function"""
    print("🎵 Welcome to Spotify Genre Segmentation Project!")
    print()
    
    # Check if Spotify API credentials are set
    if (Config.SPOTIPY_CLIENT_ID == 'your_client_id_here' or 
        Config.SPOTIPY_CLIENT_SECRET == 'your_client_secret_here'):
        print("⚠️ WARNING: Spotify API credentials not configured!")
        print("Please update config.py with your Spotify API credentials.")
        print("You can get them from: https://developer.spotify.com/dashboard")
        print()
        
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Initialize and run pipeline
    pipeline = SpotifyProjectPipeline()
    results = pipeline.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()
