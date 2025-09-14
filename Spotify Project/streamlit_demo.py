import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Spotify Genre Segmentation Demo",
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
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample data for demonstration"""
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
            popularity = np.random.randint(20, 100)
            
            data.append({
                'track_id': track_id,
                'track_name': f"{track} {i+1}",
                'artist_name': artist,
                'genre': genre,
                'danceability': danceability,
                'energy': energy,
                'valence': valence,
                'tempo': tempo,
                'loudness': loudness,
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'popularity': popularity
            })
            track_id += 1
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    """Load and cache the sample data"""
    return create_sample_data()

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Spotify Genre Segmentation & Recommendation System</h1>
        <p>Advanced Music Recommendation System using Machine Learning - DEMO VERSION</p>
    </div>
    """, unsafe_allow_html=True)

def display_statistics(df):
    """Display dataset statistics"""
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tracks", f"{len(df):,}")
    
    with col2:
        st.metric("Total Artists", f"{df['artist_name'].nunique():,}")
    
    with col3:
        st.metric("Total Genres", df['genre'].nunique())
    
    with col4:
        avg_pop = df['popularity'].mean()
        st.metric("Avg Popularity", f"{avg_pop:.1f}")

def display_genre_distribution(df):
    """Display genre distribution"""
    st.subheader("🎭 Genre Distribution")
    
    genre_counts = df['genre'].value_counts()
    
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

def display_audio_features_analysis(df):
    """Display audio features analysis"""
    st.subheader("🎼 Audio Features Analysis")
    
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
                     'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
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
            go.Histogram(x=df[feature], name=feature, showlegend=False),
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
            df.melt(
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

def display_correlation_analysis(df):
    """Display correlation analysis"""
    st.subheader("🔗 Correlation Analysis")
    
    audio_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
                     'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    corr_matrix = df[audio_features].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Audio Features Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_recommendation_demo(df):
    """Display recommendation demo"""
    st.subheader("🎯 Recommendation System Demo")
    
    # Select a track
    selected_track = st.selectbox(
        "Select a track to get recommendations:",
        df['track_name'].unique()
    )
    
    if selected_track:
        track_data = df[df['track_name'] == selected_track].iloc[0]
        
        st.write(f"**Selected Track:** {track_data['track_name']} by {track_data['artist_name']}")
        st.write(f"**Genre:** {track_data['genre']}")
        st.write(f"**Popularity:** {track_data['popularity']}")
        
        # Simple content-based recommendations using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        # Get features for selected track
        track_features = track_data[features].values.reshape(1, -1)
        
        # Get features for all other tracks
        other_tracks = df[df['track_name'] != selected_track]
        other_features = other_tracks[features].values
        
        # Calculate similarities
        similarities = cosine_similarity(track_features, other_features)[0]
        
        # Get top similar tracks
        top_indices = np.argsort(similarities)[::-1][:10]
        
        st.subheader("🎵 Top 10 Similar Tracks")
        
        for i, idx in enumerate(top_indices, 1):
            similar_track = other_tracks.iloc[idx]
            similarity_score = similarities[idx]
            
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            
            with col1:
                st.write(f"**#{i}**")
            
            with col2:
                st.write(f"**{similar_track['track_name']}**")
                st.write(f"*{similar_track['artist_name']}*")
            
            with col3:
                st.write(f"**{similar_track['genre']}**")
            
            with col4:
                st.write(f"**{similarity_score:.3f}**")
            
            st.divider()

def display_genre_explorer(df):
    """Display genre explorer"""
    st.subheader("🎭 Genre Explorer")
    
    genres = df['genre'].unique()
    selected_genre = st.selectbox("Select a genre to explore:", genres)
    
    if selected_genre:
        genre_tracks = df[df['genre'] == selected_genre]
        
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
        st.subheader("Top Tracks by Popularity")
        top_tracks = genre_tracks.nlargest(10, 'popularity')[['track_name', 'artist_name', 'popularity']]
        st.dataframe(top_tracks, use_container_width=True)

def main():
    """Main function"""
    display_header()
    
    # Load data
    df = load_data()
    
    # Sidebar navigation
    st.sidebar.title("🎵 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Audio Features", "Correlation Analysis", "Recommendations", "Genre Explorer"]
    )
    
    if page == "Overview":
        display_statistics(df)
        display_genre_distribution(df)
    
    elif page == "Audio Features":
        display_audio_features_analysis(df)
    
    elif page == "Correlation Analysis":
        display_correlation_analysis(df)
    
    elif page == "Recommendations":
        display_recommendation_demo(df)
    
    elif page == "Genre Explorer":
        display_genre_explorer(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎵 Spotify Genre Segmentation & Recommendation System - DEMO</p>
        <p>Built with ❤️ using Python, Scikit-learn, and Streamlit</p>
        <p><strong>Note:</strong> This is a demo with sample data. For real Spotify data, set up API credentials.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
