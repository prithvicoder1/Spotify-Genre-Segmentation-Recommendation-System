# 🎵 Spotify Songs' Genre Segmentation & Recommendation System

A comprehensive machine learning project that analyzes Spotify music data to perform genre segmentation and build an intelligent music recommendation system.

## 📋 Project Overview

This project implements a sophisticated music recommendation system using machine learning techniques including:

- **Data Collection**: Automated data gathering from Spotify API
- **Data Preprocessing**: Feature engineering and data cleaning
- **Exploratory Data Analysis**: Comprehensive visualizations and insights
- **Clustering Analysis**: Multiple clustering algorithms for genre segmentation
- **Hybrid Recommendation System**: Content-based + Cluster-based recommendations
- **Web Applications**: Flask and Streamlit interfaces

## 🎯 Project Objectives

1. **Perform data pre-processing operations** on Spotify music data
2. **Create comprehensive visualizations** to derive meaningful insights
3. **Generate correlation matrix** of audio features
4. **Build clustering models** for genre segmentation
5. **Develop recommendation system** based on clustering results

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Spotify Developer Account
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd spotify-recommendation-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Spotify API credentials**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Copy your Client ID and Client Secret
   - Update `config.py` with your credentials:
   ```python
   SPOTIPY_CLIENT_ID = 'your_client_id_here'
   SPOTIPY_CLIENT_SECRET = 'your_client_secret_here'
   ```

### Running the Complete Pipeline

Execute the complete ML pipeline with a single command:

```bash
python spotify_recommendation_project.py
```

This will run all steps automatically:
1. Data collection from Spotify API
2. Data preprocessing and feature engineering
3. Exploratory data analysis with visualizations
4. Clustering model training and evaluation
5. Recommendation system testing

## 📊 Project Structure

```
spotify-recommendation-project/
├── 📁 data/                          # Data storage
│   ├── spotify_data.csv             # Raw collected data
│   ├── processed_full_data.csv      # Preprocessed data
│   ├── processed_features.csv       # Feature matrix
│   └── processed_target.csv         # Target labels
├── 📁 models/                        # Trained models
│   ├── kmeans_model.pkl            # K-Means clustering model
│   ├── hierarchical_model.pkl      # Hierarchical clustering model
│   ├── gmm_model.pkl               # Gaussian Mixture Model
│   └── dbscan_model.pkl            # DBSCAN clustering model
├── 📁 plots/                         # Generated visualizations
│   ├── genre_distribution.png      # Genre distribution plots
│   ├── audio_features_analysis.png # Audio features analysis
│   ├── correlation_matrix.png      # Feature correlation matrix
│   └── clustering_visualizations/  # Clustering results
├── 📁 templates/                     # Flask HTML templates
│   └── index.html                  # Main web interface
├── 📄 config.py                     # Configuration settings
├── 📄 data_collector.py             # Spotify API data collection
├── 📄 data_preprocessing.py         # Data preprocessing pipeline
├── 📄 eda_visualization.py          # Exploratory data analysis
├── 📄 clustering_models.py          # Clustering algorithms
├── 📄 recommendation_system.py      # Hybrid recommendation system
├── 📄 flask_app.py                  # Flask web application
├── 📄 streamlit_app.py              # Streamlit web application
├── 📄 spotify_recommendation_project.py # Main pipeline script
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # This file
```

## 🔧 Individual Components

### 1. Data Collection (`data_collector.py`)

Collects music data from Spotify API including:
- Track metadata (name, artist, album, popularity)
- Audio features (danceability, energy, valence, etc.)
- Genre information from curated playlists

```python
from data_collector import SpotifyDataCollector

collector = SpotifyDataCollector()
df = collector.collect_genre_data(SAMPLE_PLAYLISTS)
```

### 2. Data Preprocessing (`data_preprocessing.py`)

Handles data cleaning and feature engineering:
- Missing value imputation
- Feature scaling and normalization
- Categorical encoding
- Feature interaction creation

```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df, X_scaled, y = preprocessor.preprocess_pipeline('data/spotify_data.csv')
```

### 3. Exploratory Data Analysis (`eda_visualization.py`)

Comprehensive data analysis and visualization:
- Genre distribution analysis
- Audio features distribution
- Correlation analysis
- Temporal analysis
- Interactive visualizations

```python
from eda_visualization import EDAAnalyzer

analyzer = EDAAnalyzer()
df, corr_matrix, genre_stats, summary = analyzer.run_complete_eda('data/processed_full_data.csv')
```

### 4. Clustering Models (`clustering_models.py`)

Multiple clustering algorithms for genre segmentation:
- K-Means Clustering
- Hierarchical Clustering
- Gaussian Mixture Model
- DBSCAN
- Model evaluation and comparison

```python
from clustering_models import ClusteringAnalyzer

analyzer = ClusteringAnalyzer()
results = analyzer.run_complete_clustering_analysis('data/processed_features.csv')
```

### 5. Recommendation System (`recommendation_system.py`)

Hybrid recommendation system combining:
- Content-based filtering
- Cluster-based recommendations
- Weighted scoring algorithm
- Multiple recommendation types

```python
from recommendation_system import SpotifyRecommendationSystem

rec_system = SpotifyRecommendationSystem()
recommendations = rec_system.get_hybrid_recommendations("Shape of You", "Ed Sheeran")
```

## 🌐 Web Applications

### Flask Application

Run the Flask web interface:

```bash
python flask_app.py
```

Visit: `http://localhost:5000`

Features:
- Interactive recommendation interface
- Real-time track search
- Multiple recommendation types
- Recommendation quality analysis

### Streamlit Application

Run the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Features:
- Interactive data exploration
- Real-time visualizations
- Genre explorer
- Clustering analysis viewer
- Recommendation interface

## 📈 Key Features

### Data Analysis
- **Genre Distribution**: Comprehensive analysis of music genres
- **Audio Features**: Statistical analysis of 13 audio features
- **Correlation Analysis**: Feature relationships and dependencies
- **Temporal Analysis**: Music trends over time

### Clustering Analysis
- **Multiple Algorithms**: K-Means, Hierarchical, GMM, DBSCAN
- **Optimal Cluster Selection**: Elbow method, Silhouette analysis
- **Visualization**: PCA and t-SNE projections
- **Performance Metrics**: Comprehensive model evaluation

### Recommendation System
- **Hybrid Approach**: Combines content-based and cluster-based methods
- **Multiple Types**: Content, cluster, diverse, and hybrid recommendations
- **Quality Analysis**: Recommendation evaluation metrics
- **Real-time Search**: Live track search and recommendations

## 🎭 Supported Genres

The system analyzes the following music genres:
- Pop
- Rock
- Hip Hop
- Electronic
- Jazz
- Classical
- Country
- R&B

## 📊 Audio Features Analyzed

- **Danceability**: How suitable a track is for dancing
- **Energy**: Perceptual measure of intensity and power
- **Valence**: Musical positivity conveyed by a track
- **Loudness**: Overall loudness of a track in decibels
- **Speechiness**: Presence of spoken words in a track
- **Acousticness**: Confidence measure of whether the track is acoustic
- **Instrumentalness**: Predicts whether a track contains no vocals
- **Liveness**: Detects the presence of an audience in the recording
- **Tempo**: Overall estimated tempo of a track in BPM
- **Key**: Key the track is in (0-11)
- **Mode**: Major (1) or minor (0)
- **Time Signature**: Estimated overall time signature

## 🔬 Machine Learning Models

### Clustering Algorithms

1. **K-Means Clustering**
   - Fast and efficient
   - Good for spherical clusters
   - Optimal for large datasets

2. **Hierarchical Clustering**
   - Creates cluster hierarchy
   - No need to specify number of clusters
   - Good for small to medium datasets

3. **Gaussian Mixture Model**
   - Probabilistic clustering
   - Handles overlapping clusters
   - Provides cluster probabilities

4. **DBSCAN**
   - Density-based clustering
   - Finds clusters of arbitrary shape
   - Identifies noise points

### Recommendation Algorithms

1. **Content-Based Filtering**
   - Uses audio feature similarity
   - Cosine similarity for feature comparison
   - Good for finding similar tracks

2. **Cluster-Based Filtering**
   - Uses clustering results
   - Recommends tracks from same cluster
   - Good for genre-based recommendations

3. **Hybrid Approach**
   - Combines content and cluster methods
   - Weighted scoring system
   - Best overall performance

## 📊 Performance Metrics

### Clustering Evaluation
- **Silhouette Score**: Measures cluster quality (-1 to 1)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Score**: Average similarity ratio of clusters

### Recommendation Evaluation
- **Precision**: Accuracy of recommendations
- **Recall**: Coverage of relevant items
- **Diversity**: Variety in recommendations
- **Popularity**: Average popularity of recommended tracks

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Spotify API Configuration
SPOTIPY_CLIENT_ID = 'your_client_id'
SPOTIPY_CLIENT_SECRET = 'your_client_secret'

# Model Configuration
N_CLUSTERS = 8
RANDOM_STATE = 42

# Recommendation Configuration
TOP_N_RECOMMENDATIONS = 10
CONTENT_WEIGHT = 0.7
CLUSTER_WEIGHT = 0.3
```

## 📝 Usage Examples

### Get Recommendations

```python
from recommendation_system import SpotifyRecommendationSystem

# Initialize system
rec_system = SpotifyRecommendationSystem()

# Get hybrid recommendations
recommendations = rec_system.get_hybrid_recommendations(
    track_name="Shape of You",
    artist_name="Ed Sheeran",
    top_n=10
)

# Display results
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['track_name']} - {rec['artist_name']}")
    print(f"   Genre: {rec['genre']}, Score: {rec['hybrid_score']:.3f}")
```

### Analyze Clusters

```python
from clustering_models import ClusteringAnalyzer

# Initialize analyzer
analyzer = ClusteringAnalyzer()

# Find optimal clusters
optimal_k, metrics = analyzer.find_optimal_clusters(X)

# Train K-Means model
model, labels = analyzer.train_kmeans(X, n_clusters=optimal_k)

# Visualize clusters
analyzer.visualize_clusters(X, labels, "K-Means")
```

## 🐛 Troubleshooting

### Common Issues

1. **Spotify API Errors**
   - Check your API credentials
   - Ensure your app has the correct permissions
   - Verify your quota limits

2. **Data Collection Issues**
   - Check internet connection
   - Verify playlist IDs are correct
   - Ensure API rate limits are not exceeded

3. **Model Training Errors**
   - Check if data is properly preprocessed
   - Verify feature dimensions
   - Ensure sufficient data for clustering

4. **Web App Issues**
   - Check if all dependencies are installed
   - Verify port availability (5000 for Flask, 8501 for Streamlit)
   - Check browser console for JavaScript errors

### Getting Help

1. Check the error messages in the console
2. Verify your configuration in `config.py`
3. Ensure all dependencies are installed correctly
4. Check the generated log files for detailed error information

## 📚 Dependencies

### Core Libraries
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `matplotlib`: Static plotting
- `seaborn`: Statistical data visualization
- `plotly`: Interactive visualizations

### API and Web
- `spotipy`: Spotify Web API wrapper
- `flask`: Web framework
- `streamlit`: Data app framework
- `requests`: HTTP library

### Utilities
- `python-dotenv`: Environment variable management
- `joblib`: Model serialization
- `jupyter`: Interactive notebooks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Spotify for providing the Web API
- The open-source community for the amazing libraries
- Music data researchers for inspiration

## 📞 Contact

For questions or support, please contact:
- Email: vijayprithvi38@gmail.com
- GitHub: prithvicoder1

---

**🎵 Happy Music Discovery! 🎵**
