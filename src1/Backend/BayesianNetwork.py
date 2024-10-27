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


# Manually define data for 10 households, ensuring they are close in terms of latitude/longitude
data = {
    'Electricity': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes'],
    'Water': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
    'Food': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes'],
    'Scale-Severity': [5, 3, 8, 2, 7, 9, 4, 6, 10, 1],
    'House': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],
    'Temporary Shelter': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Injuries': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Latitude': [40.001, 40.002, 40.003, 40.004, 40.005, 40.0015, 40.0025, 40.0035, 40.0045, 40.0055],
    'Longitude': [-75.001, -75.002, -75.003, -75.004, -75.005, -75.0015, -75.0025, -75.0035, -75.0045, -75.0055]
}

# Create a DataFrame
df_households = pd.DataFrame(data, index=[f'Household {i+1}' for i in range(10)])

# Step 2: Find neighbors within a 2km radius
def find_neighbors(df, max_distance=2):
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

print(find_neighbors(df_households, max_distance=2))
neighbors = find_neighbors(df_households, max_distance=2)

# Setup Bayesian Network structure without connecting households to each other
from pgmpy.models import BayesianNetwork

def setup_bayesian_network(neighbors):
    edges = []

    # Internal household dependencies (same as before)
    for household in neighbors.keys():
        edges.extend([(f'{household}_Electricity', f'{household}_House'),
                      (f'{household}_Water', f'{household}_House'),
                      (f'{household}_Food', f'{household}_House'),
                      (f'{household}_House', f'{household}_Temporary_Shelter'),
                      (f'{household}_Injuries', f'{household}_House')])

    # Cross-household dependencies (between neighbors) - make it unidirectional
    for household, neighbor_list in neighbors.items():
        for neighbor in neighbor_list:
            # Only add one directional dependency to avoid loops
            if int(household.split()[1]) < int(neighbor.split()[1]):  # Ensure no reverse edges
                edges.append((f'{household}_Electricity', f'{neighbor}_Electricity'))
                edges.append((f'{household}_Water', f'{neighbor}_Water'))
                edges.append((f'{household}_Food', f'{neighbor}_Food'))
                # You can add more cross-household dependencies if needed

    # Create the Bayesian Network with the edges
    model = BayesianNetwork(edges)
    print(model.nodes)
    return model


model = setup_bayesian_network(neighbors)
print(setup_bayesian_network(neighbors))

print(df_households)


# Step 4: Learn CPTs from the observed data using Maximum Likelihood Estimation
def learn_cpts(model, df):
    # Ensure that the data frame columns are properly encoded for all households and variables
    # For example, df should contain columns like 'Household 1_Electricity', 'Household 2_Water', etc.

    # Use MLE to learn the CPTs from the data
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

# Convert categorical columns to strings (use the correct column names based on your DataFrame)
categorical_columns = [
    'Electricity', 'Water', 'Food', 'House', 'Temporary Shelter', 'Injuries'
]

# Convert all categorical columns to string to avoid mixed data types
df_households[categorical_columns] = df_households[categorical_columns].astype(str)

# Ensure numerical columns are in the correct format (float or int)
numerical_columns = ['Scale-Severity', 'Latitude', 'Longitude']
df_households[numerical_columns] = df_households[numerical_columns].apply(pd.to_numeric)

# Reorganize the DataFrame into the structure required by the Bayesian network
df_reorganized = df_households.stack().reset_index(drop=True).to_frame()

# Check data types to ensure no mixed types
print(df_households.dtypes)

# Apply the learned CPTs
print(learn_cpts(model, df_reorganized))


# # Step 5: Perform inference to predict missing values for a household
# def perform_inference(model, df, household_node):
#     inference = VariableElimination(model)
#
#     # Get neighbors for the specified household
#     neighbors_list = neighbors[household_node]
#
#     # Select evidence based on neighbors' 'House' values from the dataframe
#     evidence = {}
#     for neighbor in neighbors_list:
#         evidence[neighbor] = df.loc[neighbor, 'House']  # Use household name as key
#
#     # Predict missing values for the 'House' attribute of the specified household
#     result = inference.query(variables=['House'], evidence=evidence)
#
#     print(result)
#
# # Example of how to implement the interpolation process
# df_households = simulate_data_with_missing(10)
#
# # Encode categorical variables into numerical format
# label_encoder = LabelEncoder()
#
# for column in ['Electricity', 'Water', 'Food', 'House', 'Temporary Shelter', 'Injuries']:
#     df_households[column] = label_encoder.fit_transform(df_households[column].astype(str))
#
# neighbors = find_neighbors(df_households, max_distance=3)
#
# # Setup Bayesian network with household names
# model = setup_bayesian_network(neighbors)
# model = learn_cpts(model, df_households)
#
# # Perform inference for a specific household node, e.g., 'Household 1'
# perform_inference(model, df_households, household_node='Household 1')




# Step 1: Simulate the data (same as before but with some missing data)
# def simulate_data_with_missing(num_households=10):
#     np.random.seed(42)
#     household_names = [f'Household {i+1}' for i in range(num_households)]
#     hazards = ['Flooded roads', 'Fallen trees', 'Damaged buildings', 'Power lines down', 'Blocked exits or roads']
#
#     data = {
#         'Electricity': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
#         'Water': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
#         'Food': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
#         'Scale-Severity': np.random.randint(1, 11, num_households),
#         'House': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
#         'Temporary Shelter': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
#         'Injuries': np.random.choice(['Yes', 'No', np.nan], num_households, p=[0.45, 0.45, 0.1]),
#         'Visible Hazards': [np.random.choice(hazards, size=np.random.randint(1, 4), replace=False) for _ in range(num_households)],
#         'Latitude': np.random.uniform(-90, 90, num_households),
#         'Longitude': np.random.uniform(-180, 180, num_households)
#     }
#
#     df_households = pd.DataFrame(data, index=household_names)
# # Apply one-hot encoding to Visible Hazards
# hazards_df = pd.get_dummies(df_households['Visible Hazards'].apply(pd.Series).stack()).groupby(level=0).sum()
# df_households = pd.concat([df_households.drop(columns='Visible Hazards'), hazards_df], axis=1)
#
# return df_households
