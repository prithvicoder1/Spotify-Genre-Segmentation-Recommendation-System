import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from config import Config

class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.categorical_columns = ['key', 'mode', 'time_signature']
        self.numerical_columns = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'popularity'
        ]
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            
            # Fill missing values
            # For numerical columns, use median
            for col in self.numerical_columns:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # For categorical columns, use mode
            for col in self.categorical_columns:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            # For text columns, use 'Unknown'
            text_columns = ['track_name', 'artist_name', 'album_name', 'genre']
            for col in text_columns:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna('Unknown', inplace=True)
        else:
            print("No missing values found.")
        
        return df
    
    def feature_engineering(self, df):
        """Create new features from existing ones"""
        print("Performing feature engineering...")
        
        # Convert duration from milliseconds to minutes
        if 'duration_ms' in df.columns:
            df['duration_minutes'] = df['duration_ms'] / 60000
        
        # Create energy-danceability interaction
        if 'energy' in df.columns and 'danceability' in df.columns:
            df['energy_danceability'] = df['energy'] * df['danceability']
        
        # Create valence-energy interaction
        if 'valence' in df.columns and 'energy' in df.columns:
            df['valence_energy'] = df['valence'] * df['energy']
        
        # Create acousticness-instrumentalness interaction
        if 'acousticness' in df.columns and 'instrumentalness' in df.columns:
            df['acoustic_instrumental'] = df['acousticness'] * df['instrumentalness']
        
        # Create tempo categories
        if 'tempo' in df.columns:
            df['tempo_category'] = pd.cut(df['tempo'], 
                                        bins=[0, 80, 120, 160, 200, 300], 
                                        labels=['Very Slow', 'Slow', 'Medium', 'Fast', 'Very Fast'])
        
        # Create loudness categories
        if 'loudness' in df.columns:
            df['loudness_category'] = pd.cut(df['loudness'], 
                                           bins=[-60, -20, -10, -5, 0], 
                                           labels=['Very Quiet', 'Quiet', 'Medium', 'Loud'])
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        # Encode genre
        if 'genre' in df.columns:
            df['genre_encoded'] = self.label_encoder.fit_transform(df['genre'])
        
        # One-hot encode other categorical features
        categorical_features = ['key', 'mode', 'time_signature', 'tempo_category', 'loudness_category']
        for feature in categorical_features:
            if feature in df.columns:
                dummies = pd.get_dummies(df[feature], prefix=feature)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def select_features(self, df):
        """Select relevant features for clustering"""
        print("Selecting features for clustering...")
        
        # Define feature columns for clustering
        self.feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'tempo', 'popularity', 'energy_danceability', 'valence_energy',
            'acoustic_instrumental'
        ]
        
        # Add one-hot encoded categorical features
        categorical_features = ['key', 'mode', 'time_signature', 'tempo_category', 'loudness_category']
        for feature in categorical_features:
            feature_cols = [col for col in df.columns if col.startswith(feature + '_')]
            self.feature_columns.extend(feature_cols)
        
        # Filter to only existing columns
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        print(f"Selected {len(self.feature_columns)} features for clustering")
        return df[self.feature_columns]
    
    def scale_features(self, X):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def split_data(self, X, y=None, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE):
        """Split data into train and test sets"""
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            return X_train, X_test
    
    def preprocess_pipeline(self, filepath):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Load data
        df = self.load_data(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Select features
        X = self.select_features(df)
        
        # Scale features
        X_scaled = self.scale_features(X)
        
        # Get target variable (genre)
        y = df['genre_encoded'] if 'genre_encoded' in df.columns else None
        
        print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        
        return df, X_scaled, y
    
    def save_processed_data(self, df, X_scaled, y, filepath_prefix='processed'):
        """Save processed data"""
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        
        # Save full processed dataset
        df.to_csv(os.path.join(Config.DATA_DIR, f'{filepath_prefix}_full_data.csv'), index=False)
        
        # Save scaled features
        X_scaled.to_csv(os.path.join(Config.DATA_DIR, f'{filepath_prefix}_features.csv'), index=False)
        
        # Save target if available
        if y is not None:
            pd.DataFrame(y, columns=['genre_encoded']).to_csv(
                os.path.join(Config.DATA_DIR, f'{filepath_prefix}_target.csv'), index=False
            )
        
        print("Processed data saved successfully.")

def main():
    """Main function to run preprocessing"""
    preprocessor = DataPreprocessor()
    
    # Check if data file exists
    data_file = os.path.join(Config.DATA_DIR, 'spotify_data.csv')
    
    if not os.path.exists(data_file):
        print("Data file not found. Please run data_collector.py first.")
        return
    
    # Run preprocessing pipeline
    df, X_scaled, y = preprocessor.preprocess_pipeline(data_file)
    
    # Save processed data
    preprocessor.save_processed_data(df, X_scaled, y)
    
    return df, X_scaled, y

if __name__ == "__main__":
    df, X_scaled, y = main()
