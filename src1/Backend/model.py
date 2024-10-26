import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Step 1: Define the hazard values
hazard_values = {
    'Flooded roads': 2,
    'Fallen trees': 2,
    'Damaged buildings': 5,
    'Power lines down': 4,
    'Blocked exits or roads': 3
}

# Step 2: Generate the data with multiple hazards
def simulate_data(num_households=5):
    np.random.seed(42)
    household_names = [f'Household {i+1}' for i in range(num_households)]

    # Allow multiple hazards to be selected by randomly choosing 1-3 hazards per household
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
        'User\'s Contact Information': np.random.choice(['123-456-7890', 'example@mail.com', '456-789-0123', 'contact@domain.com', '789-012-3456'], num_households)
    }

    df_households = pd.DataFrame(data, index=household_names)

    # Calculate the hazard score by summing the values of the selected hazards
    df_households['Hazard Score'] = df_households['Visible Hazards'].apply(lambda hazards: sum(hazard_values[hazard] for hazard in hazards))

    return df_households

# Step 3: Preprocess the data for KMeans (updated to include 'Visible Hazard' for clustering)
def preprocess_data(df):
    # Encode categorical columns (Yes/No) as numbers, including 'Visible Hazard' now
    label_encoder = LabelEncoder()
    for column in ['Electricity', 'Water', 'Food', 'House', 'Temporary Shelter', 'Injuries']:
        df[column] = label_encoder.fit_transform(df[column])

    # Drop 'Type of Disaster' and 'User\'s Contact Information', but keep 'Hazard Score'
    df = df.drop(['User\'s Contact Information', 'Type of Disaster'], axis=1)

    return df

# Step 4: Apply KMeans clustering
def apply_kmeans(df, num_clusters=8):
    # Initialize KMeans with a number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit the KMeans model
    clusters = kmeans.fit_predict(df)

    # Add the cluster assignments to the DataFrame
    df['Cluster'] = clusters
    return df


# Simulate the data
df_households = simulate_data()


# Preprocess the data
# (with 'Type of Disaster' removed and 'Hazard Score' included)
df_preprocessed = preprocess_data(df_households.copy())

# Apply KMeans clustering
df_clustered = apply_kmeans(df_preprocessed)

# Display the results
print(df_clustered)
