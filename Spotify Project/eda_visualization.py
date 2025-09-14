import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from config import Config

class EDAAnalyzer:
    def __init__(self):
        """Initialize EDA analyzer"""
        self.plots_dir = Config.PLOTS_DIR
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self, filepath):
        """Load processed data"""
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        return df
    
    def basic_info(self, df):
        """Display basic information about the dataset"""
        print("=" * 50)
        print("BASIC DATASET INFORMATION")
        print("=" * 50)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Number of tracks: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found.")
        
        print("\nBasic statistics:")
        print(df.describe())
        
        return df.info()
    
    def genre_distribution(self, df):
        """Analyze genre distribution"""
        print("\n" + "=" * 50)
        print("GENRE DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        genre_counts = df['genre'].value_counts()
        print("Genre distribution:")
        print(genre_counts)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot
        genre_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Genre Distribution (Bar Chart)')
        axes[0, 0].set_xlabel('Genre')
        axes[0, 0].set_ylabel('Number of Tracks')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[0, 1].pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Genre Distribution (Pie Chart)')
        
        # Horizontal bar plot
        genre_counts.plot(kind='barh', ax=axes[1, 0], color='lightcoral')
        axes[1, 0].set_title('Genre Distribution (Horizontal Bar)')
        axes[1, 0].set_xlabel('Number of Tracks')
        
        # Donut chart
        axes[1, 1].pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', 
                      wedgeprops=dict(width=0.5))
        axes[1, 1].set_title('Genre Distribution (Donut Chart)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'genre_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return genre_counts
    
    def audio_features_analysis(self, df):
        """Analyze audio features distribution"""
        print("\n" + "=" * 50)
        print("AUDIO FEATURES ANALYSIS")
        print("=" * 50)
        
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Filter existing features
        audio_features = [col for col in audio_features if col in df.columns]
        
        # Create subplots
        n_features = len(audio_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(audio_features):
            if i < len(axes):
                # Histogram
                axes[i].hist(df[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {feature.title()}')
                axes[i].set_xlabel(feature.title())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(audio_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'audio_features_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Box plots by genre
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(audio_features):
            if i < len(axes):
                df.boxplot(column=feature, by='genre', ax=axes[i])
                axes[i].set_title(f'{feature.title()} by Genre')
                axes[i].set_xlabel('Genre')
                axes[i].set_ylabel(feature.title())
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(audio_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Audio Features Distribution by Genre', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'audio_features_by_genre.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self, df):
        """Perform correlation analysis"""
        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Select numerical features for correlation
        numerical_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity'
        ]
        
        # Filter existing features
        numerical_features = [col for col in numerical_features if col in df.columns]
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_features].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Audio Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_value
                    ))
        
        if high_corr_pairs:
            print("Highly correlated feature pairs (|correlation| > 0.7):")
            for pair in high_corr_pairs:
                print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
        else:
            print("No highly correlated feature pairs found.")
        
        return corr_matrix
    
    def genre_characteristics(self, df):
        """Analyze characteristics of each genre"""
        print("\n" + "=" * 50)
        print("GENRE CHARACTERISTICS ANALYSIS")
        print("=" * 50)
        
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Filter existing features
        audio_features = [col for col in audio_features if col in df.columns]
        
        # Calculate mean values by genre
        genre_stats = df.groupby('genre')[audio_features].mean()
        
        print("Average audio features by genre:")
        print(genre_stats.round(3))
        
        # Create radar chart for each genre
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        genres = df['genre'].unique()
        
        for i, genre in enumerate(genres):
            if i < len(axes):
                values = genre_stats.loc[genre].values
                values = np.append(values, values[0])  # Close the radar chart
                
                angles = np.linspace(0, 2 * np.pi, len(audio_features), endpoint=False)
                angles = np.append(angles, angles[0])
                
                axes[i].plot(angles, values, 'o-', linewidth=2, label=genre)
                axes[i].fill(angles, values, alpha=0.25)
                axes[i].set_xticks(angles[:-1])
                axes[i].set_xticklabels(audio_features, rotation=45)
                axes[i].set_title(f'{genre.title()} Characteristics')
                axes[i].grid(True)
        
        # Hide empty subplots
        for i in range(len(genres), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'genre_characteristics_radar.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return genre_stats
    
    def popularity_analysis(self, df):
        """Analyze popularity patterns"""
        print("\n" + "=" * 50)
        print("POPULARITY ANALYSIS")
        print("=" * 50)
        
        if 'popularity' not in df.columns:
            print("Popularity feature not found in dataset.")
            return
        
        # Popularity by genre
        plt.figure(figsize=(12, 8))
        df.boxplot(column='popularity', by='genre', figsize=(12, 8))
        plt.title('Popularity Distribution by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Popularity Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'popularity_by_genre.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Popularity vs audio features
        audio_features = ['danceability', 'energy', 'valence', 'tempo']
        audio_features = [col for col in audio_features if col in df.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(audio_features):
            if i < len(axes):
                axes[i].scatter(df[feature], df['popularity'], alpha=0.6, c=df['popularity'], cmap='viridis')
                axes[i].set_xlabel(feature.title())
                axes[i].set_ylabel('Popularity')
                axes[i].set_title(f'Popularity vs {feature.title()}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'popularity_vs_features.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Top popular tracks by genre
        print("Top 5 most popular tracks by genre:")
        for genre in df['genre'].unique():
            top_tracks = df[df['genre'] == genre].nlargest(5, 'popularity')[['track_name', 'artist_name', 'popularity']]
            print(f"\n{genre.upper()}:")
            print(top_tracks.to_string(index=False))
    
    def temporal_analysis(self, df):
        """Analyze temporal patterns in the data"""
        print("\n" + "=" * 50)
        print("TEMPORAL ANALYSIS")
        print("=" * 50)
        
        if 'release_date' not in df.columns:
            print("Release date feature not found in dataset.")
            return
        
        # Convert release_date to datetime
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        
        # Songs by release year
        plt.figure(figsize=(15, 6))
        year_counts = df['release_year'].value_counts().sort_index()
        year_counts.plot(kind='line', marker='o')
        plt.title('Number of Songs by Release Year')
        plt.xlabel('Release Year')
        plt.ylabel('Number of Songs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'songs_by_year.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Genre evolution over time
        genre_year = df.groupby(['release_year', 'genre']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(15, 8))
        for genre in genre_year.columns:
            plt.plot(genre_year.index, genre_year[genre], marker='o', label=genre, linewidth=2)
        
        plt.title('Genre Evolution Over Time')
        plt.xlabel('Release Year')
        plt.ylabel('Number of Songs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'genre_evolution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plots(self, df):
        """Create interactive plots using Plotly"""
        print("\n" + "=" * 50)
        print("CREATING INTERACTIVE PLOTS")
        print("=" * 50)
        
        # Interactive scatter plot: Energy vs Valence colored by genre
        fig = px.scatter(df, x='energy', y='valence', color='genre', 
                        hover_data=['track_name', 'artist_name', 'popularity'],
                        title='Energy vs Valence by Genre')
        fig.write_html(os.path.join(self.plots_dir, 'energy_valence_interactive.html'))
        
        # Interactive 3D scatter plot
        fig = px.scatter_3d(df, x='danceability', y='energy', z='valence', 
                           color='genre', hover_data=['track_name', 'artist_name'],
                           title='3D Audio Features by Genre')
        fig.write_html(os.path.join(self.plots_dir, '3d_features_interactive.html'))
        
        # Interactive box plot
        fig = px.box(df, x='genre', y='popularity', 
                    title='Popularity Distribution by Genre')
        fig.write_html(os.path.join(self.plots_dir, 'popularity_box_interactive.html'))
        
        print("Interactive plots saved as HTML files.")
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 50)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("=" * 50)
        
        report = {
            'dataset_info': {
                'total_tracks': len(df),
                'total_features': len(df.columns),
                'genres': df['genre'].nunique(),
                'artists': df['artist_name'].nunique(),
                'albums': df['album_name'].nunique()
            },
            'genre_distribution': df['genre'].value_counts().to_dict(),
            'audio_features_stats': df[['danceability', 'energy', 'valence', 'tempo', 'popularity']].describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Save report
        import json
        with open(os.path.join(self.plots_dir, 'eda_summary_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Summary report saved as JSON file.")
        return report
    
    def run_complete_eda(self, filepath):
        """Run complete EDA pipeline"""
        print("Starting Comprehensive Exploratory Data Analysis...")
        
        # Load data
        df = self.load_data(filepath)
        
        # Basic information
        self.basic_info(df)
        
        # Genre distribution
        self.genre_distribution(df)
        
        # Audio features analysis
        self.audio_features_analysis(df)
        
        # Correlation analysis
        corr_matrix = self.correlation_analysis(df)
        
        # Genre characteristics
        genre_stats = self.genre_characteristics(df)
        
        # Popularity analysis
        self.popularity_analysis(df)
        
        # Temporal analysis
        self.temporal_analysis(df)
        
        # Interactive plots
        self.create_interactive_plots(df)
        
        # Generate summary report
        summary_report = self.generate_summary_report(df)
        
        print("\n" + "=" * 50)
        print("EDA ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"All plots saved in: {self.plots_dir}")
        print("Check the directory for all generated visualizations.")
        
        return df, corr_matrix, genre_stats, summary_report

def main():
    """Main function to run EDA"""
    analyzer = EDAAnalyzer()
    
    # Check if processed data exists
    data_file = os.path.join(Config.DATA_DIR, 'processed_full_data.csv')
    
    if not os.path.exists(data_file):
        print("Processed data file not found. Please run data_preprocessing.py first.")
        return
    
    # Run complete EDA
    df, corr_matrix, genre_stats, summary_report = analyzer.run_complete_eda(data_file)
    
    return df, corr_matrix, genre_stats, summary_report

if __name__ == "__main__":
    df, corr_matrix, genre_stats, summary_report = main()
