import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Spotify API Configuration
    SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID', 'your_client_id_here')
    SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET', 'your_client_secret_here')
    SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI', 'http://localhost:8888/callback')
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Data Configuration
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'
    
    # Model Configuration
    N_CLUSTERS = 8
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Recommendation Configuration
    TOP_N_RECOMMENDATIONS = 10
    CONTENT_WEIGHT = 0.7
    CLUSTER_WEIGHT = 0.3
