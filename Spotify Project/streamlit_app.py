import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from recommendation_system import SpotifyRecommendationSystem
from config import Config

# Page configuration
st.set_page_config(
    page_title="Spotify Genre Segmentation & Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1db954, #191414);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1db954;
    }
    
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1db954;
    }
    
    .genre-badge {
        background: #1db954;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .score-badge {
        background: #6c757d;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommendation_system():
    """Load the recommendation system with caching"""
    try:
        return SpotifyRecommendationSystem()
    except Exception as e:
        st.error(f"Error loading recommendation system: {str(e)}")
        return None

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Spotify Genre Segmentation & Recommendation System</h1>
        <p>Advanced Music Recommendation System using Machine Learning Clustering</p>
    </div>
    """, unsafe_allow_html=True)

def display_statistics(rec_system):
    """Display dataset statistics"""
    if rec_system is None or rec_system.data is None:
        return
    
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tracks", f"{len(rec_system.data):,}")
    
    with col2:
        st.metric("Total Artists", f"{rec_system.data['artist_name'].nunique():,}")
    
    with col3:
        st.metric("Total Genres", rec_system.data['genre'].nunique())
    
    with col4:
        avg_pop = rec_system.data['popularity'].mean()
        st.metric("Avg Popularity", f"{avg_pop:.1f}")

def display_genre_distribution(rec_system):
    """Display genre distribution"""
    if rec_system is None or rec_system.data is None:
        return
    
    st.subheader("🎭 Genre Distribution")
    
    genre_counts = rec_system.data['genre'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig = px.bar(
            x=genre_counts.index, 
            y=genre_counts.values,
            title="Number of Tracks by Genre",
            labels={'x': 'Genre', 'y': 'Number of Tracks'},
            color=genre_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart
        fig = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title="Genre Distribution (Percentage)"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_audio_features_analysis(rec_system):
    """Display audio features analysis"""
    if rec_system is None or rec_system.data is None:
        return
    
    st.subheader("🎼 Audio Features Analysis")
    
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Filter existing features
    audio_features = [col for col in audio_features if col in rec_system.data.columns]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=audio_features,
        specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(3)]
    )
    
    for i, feature in enumerate(audio_features):
        row = i // 3 + 1
        col = i % 3 + 1
        
        fig.add_trace(
            go.Histogram(x=rec_system.data[feature], name=feature, showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Audio Features Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plots by genre
    st.subheader("📈 Audio Features by Genre")
    
    selected_features = st.multiselect(
        "Select features to compare:",
        audio_features,
        default=audio_features[:4]
    )
    
    if selected_features:
        fig = px.box(
            rec_system.data.melt(
                id_vars=['genre'], 
                value_vars=selected_features,
                var_name='feature',
                value_name='value'
            ),
            x='genre',
            y='value',
            color='feature',
            title="Audio Features Distribution by Genre"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def display_recommendation_interface(rec_system):
    """Display the recommendation interface"""
    st.subheader("🔍 Get Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        track_name = st.text_input("Track Name *", placeholder="Enter track name...")
        artist_name = st.text_input("Artist Name", placeholder="Enter artist name...")
    
    with col2:
        recommendation_type = st.selectbox(
            "Recommendation Type",
            ["hybrid", "content", "cluster", "diverse"],
            format_func=lambda x: {
                "hybrid": "Hybrid (Best)",
                "content": "Content-Based",
                "cluster": "Cluster-Based",
                "diverse": "Diverse"
            }[x]
        )
        top_n = st.slider("Number of Recommendations", 5, 20, 10)
    
    if st.button("🎵 Get Recommendations", type="primary"):
        if not track_name:
            st.error("Please enter a track name!")
            return
        
        with st.spinner("Finding the perfect recommendations for you..."):
            try:
                if recommendation_type == "content":
                    recommendations = rec_system.get_content_recommendations(
                        track_name, artist_name, top_n
                    )
                elif recommendation_type == "cluster":
                    recommendations = rec_system.get_cluster_recommendations(
                        track_name, artist_name, top_n
                    )
                elif recommendation_type == "diverse":
                    recommendations = rec_system.get_diverse_recommendations(
                        track_name, artist_name, top_n
                    )
                else:  # hybrid
                    recommendations = rec_system.get_hybrid_recommendations(
                        track_name, artist_name, top_n=top_n
                    )
                
                if recommendations:
                    display_recommendations(recommendations, track_name, artist_name, recommendation_type)
                    
                    # Analyze recommendation quality
                    analysis = rec_system.analyze_recommendation_quality(recommendations)
                    display_recommendation_analysis(analysis)
                else:
                    st.warning("No recommendations found. Please try a different track.")
                    
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")

def display_recommendations(recommendations, track_name, artist_name, recommendation_type):
    """Display recommendations"""
    st.subheader(f"🎯 Recommendations for '{track_name}' {f'by {artist_name}' if artist_name else ''}")
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
            
            with col1:
                st.markdown(f"**#{i}**")
            
            with col2:
                st.markdown(f"**{rec['track_name']}**")
                st.markdown(f"*{rec['artist_name']}*")
                st.markdown(f"Popularity: {rec.get('popularity', 'N/A')}")
            
            with col3:
                st.markdown(f'<span class="genre-badge">{rec["genre"]}</span>', unsafe_allow_html=True)
            
            with col4:
                score = rec.get('hybrid_score', rec.get('similarity_score', rec.get('cluster_score', 0)))
                st.markdown(f'<span class="score-badge">Score: {score:.3f}</span>', unsafe_allow_html=True)
            
            st.divider()

def display_recommendation_analysis(analysis):
    """Display recommendation analysis"""
    st.subheader("📊 Recommendation Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Recommendations", analysis['total_recommendations'])
    
    with col2:
        st.metric("Average Popularity", f"{analysis['average_popularity']:.1f}")
    
    with col3:
        if analysis.get('genre_accuracy', 0) > 0:
            st.metric("Genre Accuracy", f"{analysis['genre_accuracy']:.1%}")
    
    # Genre distribution
    if analysis['genre_distribution']:
        st.subheader("🎭 Genre Distribution in Recommendations")
        
        genre_df = pd.DataFrame(list(analysis['genre_distribution'].items()), 
                               columns=['Genre', 'Count'])
        
        fig = px.pie(genre_df, values='Count', names='Genre', 
                    title="Genre Distribution in Recommendations")
        st.plotly_chart(fig, use_container_width=True)

def display_genre_explorer(rec_system):
    """Display genre explorer"""
    st.subheader("🎭 Genre Explorer")
    
    if rec_system is None or rec_system.data is None:
        return
    
    genres = rec_system.data['genre'].unique()
    selected_genre = st.selectbox("Select a genre to explore:", genres)
    
    if selected_genre:
        genre_tracks = rec_system.data[rec_system.data['genre'] == selected_genre]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Tracks", len(genre_tracks))
            st.metric("Unique Artists", genre_tracks['artist_name'].nunique())
            st.metric("Average Popularity", f"{genre_tracks['popularity'].mean():.1f}")
        
        with col2:
            # Top artists in genre
            top_artists = genre_tracks['artist_name'].value_counts().head(10)
            st.subheader("Top Artists")
            for artist, count in top_artists.items():
                st.write(f"• {artist}: {count} tracks")
        
        # Top tracks in genre
        st.subheader("Top Tracks")
        top_tracks = genre_tracks.nlargest(10, 'popularity')[['track_name', 'artist_name', 'popularity']]
        st.dataframe(top_tracks, use_container_width=True)

def display_clustering_analysis():
    """Display clustering analysis"""
    st.subheader("🔬 Clustering Analysis")
    
    # Check if clustering results exist
    plots_dir = Config.PLOTS_DIR
    if not os.path.exists(plots_dir):
        st.warning("Clustering analysis not available. Please run clustering_models.py first.")
        return
    
    # List available clustering plots
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    if plot_files:
        st.write("Available clustering visualizations:")
        
        for plot_file in plot_files:
            if 'cluster' in plot_file.lower():
                st.image(os.path.join(plots_dir, plot_file), caption=plot_file.replace('.png', '').replace('_', ' ').title())
    else:
        st.warning("No clustering visualizations found.")

def main():
    """Main function"""
    display_header()
    
    # Load recommendation system
    rec_system = load_recommendation_system()
    
    if rec_system is None:
        st.error("""
        **Recommendation system could not be loaded!**
        
        Please ensure you have:
        1. ✅ Run `data_collector.py` to collect data
        2. ✅ Run `data_preprocessing.py` to preprocess data  
        3. ✅ Run `clustering_models.py` to train models
        4. ✅ Set up your Spotify API credentials in `config.py`
        
        Then restart this Streamlit app.
        """)
        return
    
    # Sidebar navigation
    st.sidebar.title("🎵 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Recommendations", "Genre Explorer", "Clustering Analysis"]
    )
    
    if page == "Overview":
        display_statistics(rec_system)
        display_genre_distribution(rec_system)
        display_audio_features_analysis(rec_system)
    
    elif page == "Recommendations":
        display_recommendation_interface(rec_system)
    
    elif page == "Genre Explorer":
        display_genre_explorer(rec_system)
    
    elif page == "Clustering Analysis":
        display_clustering_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎵 Spotify Genre Segmentation & Recommendation System</p>
        <p>Built with ❤️ using Python, Scikit-learn, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
