import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Define the hazard values
hazard_values = {
    'Flooded roads': 2,
    'Fallen trees': 2,
    'Damaged buildings': 5,
    'Power lines down': 4,
    'Blocked exits or roads': 3
}

# Step 2: Generate the data with multiple hazards and address column
def simulate_data(num_households=1000):
    np.random.seed(42)
    household_names = [f'Household {i+1}' for i in range(num_households)]

    hazards = list(hazard_values.keys())
    data = {
        'Electricity': np.random.choice(['Yes', 'No'], num_households),
        'Water': np.random.choice(['Yes', 'No'], num_households),
        'Food': np.random.choice(['Yes', 'No'], num_households),
        'Scale-Severity': np.random.randint(1, 11, num_households),  # Scale from 1 to 10
        'House': np.random.choice(['Yes', 'No'], num_households),
        'Temporary Shelter': np.random.choice(['Yes', 'No'], num_households),
        'Injuries': np.random.choice(['Yes', 'No'], num_households),
        'Visible Hazards': [np.random.choice(hazards, size=np.random.randint(1, 4), replace=False) for _ in range(num_households)],
        'Type of Disaster': np.random.choice(['Hurricane', 'Flood', 'Wildfire', 'Earthquake', 'Tornado'], num_households),
        'User\'s Contact Information': np.random.choice(['123-456-7890', 'example@mail.com', '456-789-0123', 'contact@domain.com', '789-012-3456'], num_households),
        'Address': [(round(np.random.uniform(-180, 180), 6), round(np.random.uniform(-90, 90), 6)) for _ in range(num_households)]  # Random Longitude, Latitude
    }

    df_households = pd.DataFrame(data, index=household_names)

    # Calculate the hazard score by summing the values of the selected hazards
    df_households['Hazard Score'] = df_households['Visible Hazards'].apply(lambda hazards: sum(hazard_values[hazard] for hazard in hazards))

    return df_households

# Step 3: One-Hot Encoding for Visible Hazards
def one_hot_encode_hazards(df):
    # One-Hot Encode hazards
    for hazard in hazard_values.keys():
        df[hazard] = df['Visible Hazards'].apply(lambda hazards: 1 if hazard in hazards else 0)

    return df

# Step 4: Feature engineering (scaling, interaction terms, damage index, etc.)
def feature_engineer(df):
    label_encoder = LabelEncoder()

    # Encode binary columns (Yes/No) as 0/1
    for column in ['Electricity', 'Water', 'Food', 'House', 'Temporary Shelter', 'Injuries']:
        df[column] = label_encoder.fit_transform(df[column])

    # One-Hot Encoding for visible hazards
    df = one_hot_encode_hazards(df)

    # Interaction terms between binary columns (to capture combined effects)
    df['Electricity & Water'] = df['Electricity'] & df['Water']
    df['Electricity & Food'] = df['Electricity'] & df['Food']
    df['Water & Food'] = df['Water'] & df['Food']

    # Composite Damage Index: Sum of all key features
    df['Damage Index'] = df['Electricity'] + df['Water'] + df['Food'] + df['House'] + df['Temporary Shelter'] + df['Hazard Score'] + df['Scale-Severity']

    # Weighted Hazard Score
    df['Weighted Hazard Score'] = df['Hazard Score'] * df['Scale-Severity']  # Weight hazard by severity

    # Binary interaction for utilities (count how many utilities are lost)
    df['Total Utility Loss'] = df[['Electricity', 'Water', 'Food']].sum(axis=1)

    # Handling outliers: Cap the Hazard Score and Scale-Severity to reasonable values
    df['Hazard Score'] = np.clip(df['Hazard Score'], 0, 15)  # Cap Hazard Score
    df['Scale-Severity'] = np.clip(df['Scale-Severity'], 1, 10)  # Cap Scale-Severity between 1 and 10

    # Scaling numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    df[['Hazard Score', 'Scale-Severity', 'Damage Index', 'Weighted Hazard Score']] = scaler.fit_transform(df[['Hazard Score', 'Scale-Severity', 'Damage Index', 'Weighted Hazard Score']])

    # Drop irrelevant columns for clustering (including Address for now)
    df = df.drop(['User\'s Contact Information', 'Type of Disaster', 'Visible Hazards', 'Address'], axis=1)

    return df

# Step 5: Apply Principal Component Analysis (PCA) for dimensionality reduction
def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)

    # Return the reduced PCA features as a DataFrame
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(df_pca, columns=pca_columns)

    return df_pca

# Step 6: Apply KMeans Clustering
def apply_kmeans(df, n_clusters=3):
    # Initialize KMeans with a specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit KMeans on the PCA-reduced dataset
    clusters = kmeans.fit_predict(df)

    # Add the cluster labels to the DataFrame
    df['Cluster'] = clusters

    return df, kmeans

# Step 7: Visualize the Clusters in 2D space
def visualize_clusters(df_pca, kmeans):
    plt.figure(figsize=(10, 6))

    # Scatter plot of the PCA components, colored by cluster
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', s=100, alpha=0.7)

    # Plot cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')

    plt.title('KMeans Clustering on PCA-Reduced Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

# Simulate the data
df_households = simulate_data()

# Apply feature engineering (One-Hot Encoding, interaction terms, scaling, etc.)
df_engineered = feature_engineer(df_households.copy())

# Apply PCA to reduce dimensionality to 2 components
df_pca = apply_pca(df_engineered, n_components=2)

# Apply KMeans clustering on the PCA-reduced data
df_clustered, kmeans_model = apply_kmeans(df_pca)

# Visualize the clusters
visualize_clusters(df_clustered, kmeans_model)

# Show the original data including the 'Address' column to be used after clustering
print(df_households[['Address', 'Hazard Score']].sort_values(by='Hazard Score', ascending=False))
