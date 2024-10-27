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

# Drop any columns not required for the Bayesian Network edges
columns_to_keep = ['Electricity', 'Water', 'Food', 'Scale-Severity', 'House', 'Temporary Shelter', 'Injuries', 'Latitude', 'Longitude']
df_households = df_households[columns_to_keep]

# Step 1: Reorganize the DataFrame to map edges (e.g., 'Household_1_Electricity')
df_reorganized = pd.DataFrame()

# Attributes for the network
attributes = ['Electricity', 'Water', 'Food', 'House', 'Temporary Shelter', 'Injuries']
numerical_attributes = ['Scale-Severity', 'Latitude', 'Longitude']

# Populate the reorganized DataFrame with both categorical and numerical values
for household in df_households.index:
    for attribute in attributes + numerical_attributes:
        column_name = f"{household.replace(' ', '_')}_{attribute.replace(' ', '_')}"
        df_reorganized[column_name] = [df_households.loc[household, attribute]] * len(df_households)

# Step 2: Find neighbors within a 2km radius
def find_neighbors(df, max_distance=2):
    neighbors = {}
    for i, row1 in df.iterrows():
        neighbors[row1.name] = []
        for j, row2 in df.iterrows():
            if i != j:
                dist = haversine(row1['Longitude'], row1['Latitude'], row2['Longitude'], row2['Latitude'])
                if dist <= max_distance:
                    neighbors[row1.name].append(row2.name)
    return neighbors

neighbors = find_neighbors(df_households)

# Step 3: Setup Bayesian Network
def setup_bayesian_network(neighbors):
    edges = []

    # Internal household dependencies
    for household in neighbors.keys():
        edges.extend([(f'{household}_Electricity', f'{household}_House'),
                      (f'{household}_Water', f'{household}_House'),
                      (f'{household}_Food', f'{household}_House'),
                      (f'{household}_House', f'{household}_Temporary_Shelter'),
                      (f'{household}_Injuries', f'{household}_House')])

    # Cross-household dependencies
    for household, neighbor_list in neighbors.items():
        for neighbor in neighbor_list:
            if int(household.split()[1]) < int(neighbor.split()[1]):
                edges.append((f'{household}_Electricity', f'{neighbor}_Electricity'))
                edges.append((f'{household}_Water', f'{neighbor}_Water'))
                edges.append((f'{household}_Food', f'{neighbor}_Food'))

    model = BayesianNetwork(edges)
    return model

model = setup_bayesian_network(neighbors)

# Step 4: Learn CPTs from the observed data using Maximum Likelihood Estimation
def learn_cpts(model, df):
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

model = learn_cpts(model, df_reorganized)

# Step 5: Perform inference to predict missing values for a household
def perform_inference(model, df, household_node):
    inference = VariableElimination(model)

    # Get neighbors for the specified household
    neighbors_list = neighbors[household_node]

    # Select evidence based on neighbors' 'House' values
    evidence = {}
    for neighbor in neighbors_list:
        evidence[f'{neighbor}_House'] = df.loc[neighbor, f'{neighbor}_House']

    # Predict missing values for the 'House' attribute of the specified household
    result = inference.query(variables=[f'{household_node}_House'], evidence=evidence)

    print(result)

# Perform inference for a specific household node, e.g., 'Household 1'
perform_inference(model, df_reorganized, household_node='Household_1')

# Print the final dataframe
print(df_reorganized)
