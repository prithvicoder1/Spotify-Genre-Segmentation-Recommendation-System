import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import Config

class ClusteringAnalyzer:
    def __init__(self):
        """Initialize clustering analyzer"""
        self.models_dir = Config.MODELS_DIR
        self.plots_dir = Config.PLOTS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.tsne = TSNE(n_components=2, random_state=Config.RANDOM_STATE)
        
        # Store trained models
        self.trained_models = {}
        self.cluster_labels = {}
        
    def load_data(self, features_file, target_file=None):
        """Load processed features and target data"""
        X = pd.read_csv(features_file)
        print(f"Loaded features with shape: {X.shape}")
        
        y = None
        if target_file and os.path.exists(target_file):
            y = pd.read_csv(target_file)
            print(f"Loaded target with shape: {y.shape}")
        
        return X, y
    
    def find_optimal_clusters(self, X, max_clusters=15):
        """Find optimal number of clusters using multiple methods"""
        print("Finding optimal number of clusters...")
        
        # Elbow method
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(X, cluster_labels))
        
        # Plot evaluation metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Elbow method
        axes[0, 0].plot(K_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True)
        
        # Silhouette score
        axes[0, 1].plot(K_range, silhouette_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score')
        axes[0, 1].grid(True)
        
        # Calinski-Harabasz score
        axes[1, 0].plot(K_range, calinski_scores, 'go-')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Score')
        axes[1, 0].grid(True)
        
        # Davies-Bouldin score
        axes[1, 1].plot(K_range, davies_bouldin_scores, 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].set_title('Davies-Bouldin Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'cluster_evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k based on silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k, {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores
        }
    
    def train_kmeans(self, X, n_clusters=None):
        """Train K-Means clustering model"""
        if n_clusters is None:
            n_clusters = Config.N_CLUSTERS
        
        print(f"Training K-Means with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        self.trained_models['kmeans'] = kmeans
        self.cluster_labels['kmeans'] = cluster_labels
        
        # Evaluate model
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        print(f"K-Means Results:")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
        
        return kmeans, cluster_labels
    
    def train_hierarchical(self, X, n_clusters=None):
        """Train Hierarchical clustering model"""
        if n_clusters is None:
            n_clusters = Config.N_CLUSTERS
        
        print(f"Training Hierarchical Clustering with {n_clusters} clusters...")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = hierarchical.fit_predict(X)
        
        self.trained_models['hierarchical'] = hierarchical
        self.cluster_labels['hierarchical'] = cluster_labels
        
        # Evaluate model
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        print(f"Hierarchical Clustering Results:")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
        
        return hierarchical, cluster_labels
    
    def train_gaussian_mixture(self, X, n_clusters=None):
        """Train Gaussian Mixture Model"""
        if n_clusters is None:
            n_clusters = Config.N_CLUSTERS
        
        print(f"Training Gaussian Mixture Model with {n_clusters} clusters...")
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=Config.RANDOM_STATE)
        cluster_labels = gmm.fit_predict(X)
        
        self.trained_models['gmm'] = gmm
        self.cluster_labels['gmm'] = cluster_labels
        
        # Evaluate model
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        print(f"Gaussian Mixture Model Results:")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
        
        return gmm, cluster_labels
    
    def train_dbscan(self, X, eps=0.5, min_samples=5):
        """Train DBSCAN clustering model"""
        print(f"Training DBSCAN with eps={eps}, min_samples={min_samples}...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        self.trained_models['dbscan'] = dbscan
        self.cluster_labels['dbscan'] = cluster_labels
        
        # Count clusters and noise points
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"DBSCAN Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        
        if n_clusters > 1:
            # Evaluate model (only if we have more than 1 cluster)
            silhouette_avg = silhouette_score(X, cluster_labels)
            calinski_score = calinski_harabasz_score(X, cluster_labels)
            davies_bouldin = davies_bouldin_score(X, cluster_labels)
            
            print(f"  Silhouette Score: {silhouette_avg:.3f}")
            print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")
            print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
        
        return dbscan, cluster_labels
    
    def visualize_clusters(self, X, cluster_labels, method_name, y=None):
        """Visualize clusters using PCA and t-SNE"""
        print(f"Visualizing {method_name} clusters...")
        
        # PCA visualization
        X_pca = self.pca.fit_transform(X)
        
        # t-SNE visualization
        X_tsne = self.tsne.fit_transform(X)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA scatter plot
        scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title(f'{method_name} - PCA Visualization')
        axes[0, 0].set_xlabel('First Principal Component')
        axes[0, 0].set_ylabel('Second Principal Component')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # t-SNE scatter plot
        scatter2 = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        axes[0, 1].set_title(f'{method_name} - t-SNE Visualization')
        axes[0, 1].set_xlabel('t-SNE Component 1')
        axes[0, 1].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # If we have true labels, show them
        if y is not None:
            # PCA with true labels
            scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y.iloc[:, 0], cmap='tab10', alpha=0.6)
            axes[1, 0].set_title('True Labels - PCA Visualization')
            axes[1, 0].set_xlabel('First Principal Component')
            axes[1, 0].set_ylabel('Second Principal Component')
            plt.colorbar(scatter3, ax=axes[1, 0])
            
            # t-SNE with true labels
            scatter4 = axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.iloc[:, 0], cmap='tab10', alpha=0.6)
            axes[1, 1].set_title('True Labels - t-SNE Visualization')
            axes[1, 1].set_xlabel('t-SNE Component 1')
            axes[1, 1].set_ylabel('t-SNE Component 2')
            plt.colorbar(scatter4, ax=axes[1, 1])
        else:
            # Hide the bottom row if no true labels
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{method_name.lower()}_clusters_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_cluster_characteristics(self, X, cluster_labels, method_name, feature_names=None):
        """Analyze characteristics of each cluster"""
        print(f"Analyzing {method_name} cluster characteristics...")
        
        # Add cluster labels to the data
        X_with_clusters = X.copy()
        X_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = X_with_clusters.groupby('cluster').agg(['mean', 'std']).round(3)
        
        print(f"\n{method_name} Cluster Statistics:")
        print(cluster_stats)
        
        # Visualize cluster characteristics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        if n_clusters > 0:
            # Select a subset of features for visualization
            if feature_names is None:
                feature_names = X.columns[:8]  # First 8 features
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(feature_names[:4]):
                if feature in X.columns:
                    cluster_means = X_with_clusters.groupby('cluster')[feature].mean()
                    cluster_means.plot(kind='bar', ax=axes[i], color='skyblue')
                    axes[i].set_title(f'{feature.title()} by Cluster')
                    axes[i].set_xlabel('Cluster')
                    axes[i].set_ylabel(feature.title())
                    axes[i].tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'{method_name.lower()}_cluster_characteristics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        return cluster_stats
    
    def compare_clustering_methods(self, X, y=None):
        """Compare different clustering methods"""
        print("Comparing clustering methods...")
        
        # Train all models
        methods = {
            'K-Means': self.train_kmeans,
            'Hierarchical': self.train_hierarchical,
            'Gaussian Mixture': self.train_gaussian_mixture,
            'DBSCAN': self.train_dbscan
        }
        
        results = {}
        
        for method_name, train_func in methods.items():
            print(f"\n{'='*50}")
            print(f"Training {method_name}")
            print('='*50)
            
            try:
                if method_name == 'DBSCAN':
                    model, labels = train_func(X)
                else:
                    model, labels = train_func(X)
                
                # Evaluate
                if len(set(labels)) > 1:  # More than 1 cluster
                    silhouette_avg = silhouette_score(X, labels)
                    calinski_score = calinski_harabasz_score(X, labels)
                    davies_bouldin = davies_bouldin_score(X, labels)
                    
                    results[method_name] = {
                        'model': model,
                        'labels': labels,
                        'silhouette_score': silhouette_avg,
                        'calinski_score': calinski_score,
                        'davies_bouldin_score': davies_bouldin,
                        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
                    }
                    
                    # Visualize
                    self.visualize_clusters(X, labels, method_name, y)
                    
                    # Analyze characteristics
                    self.analyze_cluster_characteristics(X, labels, method_name)
                    
            except Exception as e:
                print(f"Error training {method_name}: {str(e)}")
                continue
        
        # Create comparison table
        if results:
            comparison_df = pd.DataFrame({
                method: {
                    'Silhouette Score': results[method]['silhouette_score'],
                    'Calinski-Harabasz Score': results[method]['calinski_score'],
                    'Davies-Bouldin Score': results[method]['davies_bouldin_score'],
                    'Number of Clusters': results[method]['n_clusters']
                }
                for method in results.keys()
            }).T
            
            print("\n" + "="*50)
            print("CLUSTERING METHODS COMPARISON")
            print("="*50)
            print(comparison_df.round(3))
            
            # Save comparison
            comparison_df.to_csv(os.path.join(self.plots_dir, 'clustering_comparison.csv'))
            
            # Find best method based on silhouette score
            best_method = comparison_df['Silhouette Score'].idxmax()
            print(f"\nBest clustering method: {best_method}")
            print(f"Silhouette Score: {comparison_df.loc[best_method, 'Silhouette Score']:.3f}")
        
        return results
    
    def save_models(self):
        """Save trained models"""
        print("Saving trained models...")
        
        for method_name, model in self.trained_models.items():
            model_path = os.path.join(self.models_dir, f'{method_name.lower()}_model.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {method_name} model to {model_path}")
        
        # Save cluster labels
        for method_name, labels in self.cluster_labels.items():
            labels_path = os.path.join(self.models_dir, f'{method_name.lower()}_labels.pkl')
            joblib.dump(labels, labels_path)
            print(f"Saved {method_name} labels to {labels_path}")
    
    def load_models(self):
        """Load trained models"""
        print("Loading trained models...")
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            method_name = model_file.replace('_model.pkl', '').title()
            model_path = os.path.join(self.models_dir, model_file)
            
            try:
                model = joblib.load(model_path)
                self.trained_models[method_name] = model
                print(f"Loaded {method_name} model")
            except Exception as e:
                print(f"Error loading {method_name} model: {str(e)}")
    
    def run_complete_clustering_analysis(self, features_file, target_file=None):
        """Run complete clustering analysis pipeline"""
        print("Starting Complete Clustering Analysis...")
        
        # Load data
        X, y = self.load_data(features_file, target_file)
        
        # Find optimal number of clusters
        optimal_k, evaluation_metrics = self.find_optimal_clusters(X)
        
        # Update config with optimal k
        Config.N_CLUSTERS = optimal_k
        
        # Compare clustering methods
        results = self.compare_clustering_methods(X, y)
        
        # Save models
        self.save_models()
        
        print("\n" + "="*50)
        print("CLUSTERING ANALYSIS COMPLETE!")
        print("="*50)
        print(f"All models and visualizations saved in: {self.models_dir} and {self.plots_dir}")
        
        return results

def main():
    """Main function to run clustering analysis"""
    analyzer = ClusteringAnalyzer()
    
    # Check if processed data exists
    features_file = os.path.join(Config.DATA_DIR, 'processed_features.csv')
    target_file = os.path.join(Config.DATA_DIR, 'processed_target.csv')
    
    if not os.path.exists(features_file):
        print("Processed features file not found. Please run data_preprocessing.py first.")
        return
    
    # Run complete clustering analysis
    results = analyzer.run_complete_clustering_analysis(features_file, target_file)
    
    return results

if __name__ == "__main__":
    results = main()
