#!/usr/bin/env python3
"""
Setup script for Spotify Genre Segmentation Project
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    directories = ['data', 'models', 'plots', 'templates']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created: {directory}/")
    
    return True

def check_spotify_credentials():
    """Check if Spotify credentials are configured"""
    print("🔑 Checking Spotify API credentials...")
    
    try:
        from config import Config
        
        if (Config.SPOTIPY_CLIENT_ID == 'your_client_id_here' or 
            Config.SPOTIPY_CLIENT_SECRET == 'your_client_secret_here'):
            print("⚠️ Spotify API credentials not configured!")
            print("Please update config.py with your Spotify API credentials.")
            print("You can get them from: https://developer.spotify.com/dashboard")
            return False
        else:
            print("✅ Spotify API credentials configured!")
            return True
    except ImportError:
        print("❌ Could not import config.py")
        return False

def main():
    """Main setup function"""
    print("🎵 Spotify Genre Segmentation Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check Spotify credentials
    spotify_configured = check_spotify_credentials()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("=" * 50)
    
    if spotify_configured:
        print("✅ Ready to run the project!")
        print("\nNext steps:")
        print("1. Run: python spotify_recommendation_project.py")
        print("2. Or run individual components:")
        print("   - python data_collector.py")
        print("   - python data_preprocessing.py")
        print("   - python eda_visualization.py")
        print("   - python clustering_models.py")
        print("   - python recommendation_system.py")
        print("3. Run web apps:")
        print("   - python flask_app.py")
        print("   - streamlit run streamlit_app.py")
    else:
        print("⚠️ Please configure Spotify API credentials first!")
        print("1. Go to: https://developer.spotify.com/dashboard")
        print("2. Create a new app")
        print("3. Copy Client ID and Client Secret")
        print("4. Update config.py with your credentials")
        print("5. Then run: python spotify_recommendation_project.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
