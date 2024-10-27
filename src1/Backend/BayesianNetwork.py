import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

# Function to calculate distance between two locations (Haversine formula)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # Radius of the Earth in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Step 1: Simulate the data (same as before but with some missing data)
def simulate_data_with_missing(num_households=10):
    np.random.seed(42)
    household_names = [f'Household {i+1}' for i in range(num_households)]
    hazards = ['Flooded roads', 'Fallen trees', 'Damaged buildings', 'Power lines down', 'Blocked exits or roads']

    data = {
        'Electricity': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
        'Water': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
        'Food': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
        'Scale-Severity': np.random.randint(1, 11, num_households),
        'House': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
        'Temporary Shelter': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
        'Injuries': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
        'Visible Hazards': [np.random.choice(hazards, size=np.random.randint(1, 4), replace=False) for _ in range(num_households)],
        'Latitude': np.random.uniform(-90, 90, num_households),
        'Longitude': np.random.uniform(-180, 180, num_households)
    }

    df_households = pd.DataFrame(data, index=household_names)

    # Apply one-hot encoding to Visible Hazards
    hazards_df = pd.get_dummies(df_households['Visible Hazards'].apply(pd.Series).stack()).groupby(level=0).sum()
    df_households = pd.concat([df_households.drop(columns='Visible Hazards'), hazards_df], axis=1)

    return df_households

# Step 2: Find neighbors within a 3km radius
def find_neighbors(df, max_distance=3):
    neighbors = {}
    for i, row1 in df.iterrows():
        neighbors[row1.name] = []  # Use household names as keys
        for j, row2 in df.iterrows():
            if i != j:
                # Fix: Pass both lon2 and lat2 to the haversine function
                dist = haversine(row1['Longitude'], row1['Latitude'], row2['Longitude'], row2['Latitude'])
                if dist <= max_distance:
                    neighbors[row1.name].append(row2.name)  # Use household names as neighbor
    return neighbors




# Step 3: Setup Bayesian Network structure using household names
def setup_bayesian_network(neighbors):
    edges = []
    for household, neighbor_list in neighbors.items():
        for neighbor in neighbor_list:
            edges.append((neighbor, household))  # Connect neighbors (households) to each other
    return BayesianNetwork(edges)

# Step 4: Learn CPTs from the observed data using Maximum Likelihood Estimation
def learn_cpts(model, df):
    # Use MLE to learn the CPTs from the data
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

# Step 5: Perform inference to predict missing values for a household
def perform_inference(model, df, household_node):
    inference = VariableElimination(model)

    # Get neighbors for the specified household
    neighbors_list = neighbors[household_node]

    # Select evidence based on neighbors' 'House' values from the dataframe
    evidence = {}
    for neighbor in neighbors_list:
        evidence[neighbor] = df.loc[neighbor, 'House']  # Use household name as key

    # Predict missing values for the 'House' attribute of the specified household
    result = inference.query(variables=['House'], evidence=evidence)

    print(result)

# Example of how to implement the interpolation process
df_households = simulate_data_with_missing(10)

# Encode categorical variables into numerical format
label_encoder = LabelEncoder()

for column in ['Electricity', 'Water', 'Food', 'House', 'Temporary Shelter', 'Injuries']:
    df_households[column] = label_encoder.fit_transform(df_households[column].astype(str))

neighbors = find_neighbors(df_households, max_distance=3)

# Setup Bayesian network with household names
model = setup_bayesian_network(neighbors)
model = learn_cpts(model, df_households)

# Perform inference for a specific household node, e.g., 'Household 1'
perform_inference(model, df_households, household_node='Household 1')
