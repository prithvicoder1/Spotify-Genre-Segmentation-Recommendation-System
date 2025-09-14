# 🎵 Spotify Project Setup Guide

## Quick Start (Demo Version)

If you want to see the system in action immediately without Spotify API setup:

```bash
python demo_without_api.py
```

This creates sample data and demonstrates all the features!

## Full Setup with Real Spotify Data

### Step 1: Get Spotify API Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with your Spotify account
3. Click "Create App"
4. Fill in the app details:
   - App name: "Spotify Genre Segmentation"
   - App description: "Music recommendation system"
   - Website: (optional)
   - Redirect URI: `http://localhost:8888/callback`
5. Click "Save"
6. Copy your **Client ID** and **Client Secret**

### Step 2: Configure the Project

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add your credentials to the `.env` file:

```env
SPOTIPY_CLIENT_ID=your_actual_client_id_here
SPOTIPY_CLIENT_SECRET=your_actual_client_secret_here
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```

### Step 3: Run the Complete Pipeline

```bash
# Run the complete ML pipeline
python spotify_recommendation_project.py
```

This will:
1. ✅ Collect data from Spotify API
2. ✅ Preprocess and clean the data
3. ✅ Perform exploratory data analysis
4. ✅ Train clustering models
5. ✅ Test the recommendation system

### Step 4: Launch Web Applications

#### Flask Web App
```bash
python flask_app.py
```
Visit: http://localhost:5000

#### Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Auto-opens in browser

## Individual Components

You can also run individual components:

```bash
# Data collection only
python data_collector.py

# Data preprocessing only
python data_preprocessing.py

# EDA and visualizations only
python eda_visualization.py

# Clustering analysis only
python clustering_models.py

# Recommendation system only
python recommendation_system.py

# Example usage
python example_usage.py
```

## Troubleshooting

### Common Issues

1. **"No module named 'pandas'"**
   ```bash
   pip install -r requirements.txt
   ```

2. **Spotify API errors**
   - Check your credentials in `.env` file
   - Ensure your app has the correct permissions
   - Verify your quota limits

3. **Data collection issues**
   - Check internet connection
   - Verify playlist IDs are correct
   - Ensure API rate limits are not exceeded

4. **Web app issues**
   - Check if port 5000 (Flask) or 8501 (Streamlit) is available
   - Ensure all dependencies are installed

### Getting Help

1. Check error messages in the console
2. Verify your configuration
3. Ensure all dependencies are installed
4. Check the generated log files

## Project Structure

```
spotify-recommendation-project/
├── 📁 data/                    # Data storage
├── 📁 models/                  # Trained models
├── 📁 plots/                   # Generated visualizations
├── 📁 templates/               # Flask HTML templates
├── 📄 config.py               # Configuration
├── 📄 data_collector.py       # Spotify API data collection
├── 📄 data_preprocessing.py   # Data preprocessing
├── 📄 eda_visualization.py    # Exploratory data analysis
├── 📄 clustering_models.py    # Clustering algorithms
├── 📄 recommendation_system.py # Recommendation system
├── 📄 flask_app.py            # Flask web app
├── 📄 streamlit_app.py        # Streamlit dashboard
├── 📄 spotify_recommendation_project.py # Main pipeline
├── 📄 demo_without_api.py     # Demo without API
├── 📄 example_usage.py        # Usage examples
├── 📄 requirements.txt        # Dependencies
└── 📄 README.md              # Full documentation
```

## What You'll Get

### Data Analysis
- Genre distribution analysis
- Audio features visualization
- Correlation matrix
- Temporal analysis

### Clustering Results
- K-Means clustering
- Hierarchical clustering
- Gaussian Mixture Model
- DBSCAN
- Model evaluation metrics

### Recommendation System
- Content-based recommendations
- Cluster-based recommendations
- Hybrid approach
- Multiple recommendation types

### Web Interfaces
- Beautiful Flask web app
- Interactive Streamlit dashboard
- Real-time recommendations
- Data visualization

## Next Steps

1. **Run the demo**: `python demo_without_api.py`
2. **Set up Spotify API**: Follow Step 1-2 above
3. **Run full pipeline**: `python spotify_recommendation_project.py`
4. **Launch web apps**: Flask and Streamlit
5. **Explore results**: Check the plots/ directory

## Support

For questions or issues:
1. Check the README.md for detailed documentation
2. Review error messages in the console
3. Ensure all dependencies are installed correctly
4. Verify your Spotify API credentials

---

**🎵 Happy Music Discovery! 🎵**
