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
# Keep only Electricity, Water, Food, Scale-Severity, House, Temporary Shelter, Injuries, Latitude, and Longitude
columns_to_keep = ['Electricity', 'Water', 'Food', 'Scale-Severity', 'House', 'Temporary Shelter', 'Injuries', 'Latitude', 'Longitude']
df_households = df_households[columns_to_keep]

# Print the cleaned DataFrame
print("Cleaned DataFrame:")
print(df_households)

# Step 1: Initialize an empty DataFrame
df_reorganized = pd.DataFrame()

# List of attributes to transform
attributes = ['Electricity', 'Water', 'Food', 'House', 'Temporary Shelter', 'Injuries']

# Step 2: Iterate over each household and each attribute and fill the reorganized DataFrame
for household in df_households.index:
    for attribute in attributes:
        column_name = f"{household}_{attribute.replace(' ', '_')}"  # e.g., Household_1_Electricity
        # Assign the value for this household and attribute into the new DataFrame
        df_reorganized[column_name] = [df_households.loc[household, attribute]]

# Step 3: For numerical values, handle them similarly (like Scale-Severity, Latitude, Longitude)
numerical_attributes = ['Scale-Severity', 'Latitude', 'Longitude']

for household in df_households.index:
    for attribute in numerical_attributes:
        column_name = f"{household}_{attribute.replace(' ', '_')}"
        # Assign the value for this household and numerical attribute into the new DataFrame
        df_reorganized[column_name] = [df_households.loc[household, attribute]]

# Check the final DataFrame to ensure everything is properly organized
print(df_reorganized)


# Step 2: Find neighbors within a 2km radius
def find_neighbors(df, max_distance=2):
    neighbors = {}
    for i, row1 in df.iterrows():
        neighbors[row1.name] = []  # Use household names as keys
        for j, row2 in df.iterrows():
            if i != j:
                dist = haversine(row1['Longitude'], row1['Latitude'], row2['Longitude'], row2['Latitude'])
                if dist <= max_distance:
                    neighbors[row1.name].append(row2.name)
    return neighbors

neighbors = find_neighbors(df_households, max_distance=2)

# Step 3: Setup Bayesian Network
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
                edges.append((f'{household}_House', f'{neighbor}_House'))
                edges.append((f'{household}_Injuries', f'{neighbor}_Injuries'))
                edges.append((f'{household}_Temporary_Shelter', f'{neighbor}_Temporary_Shelter'))

    # Create the Bayesian Network with the defined edges
    model = BayesianNetwork(edges)
    return model

model = setup_bayesian_network(neighbors)

# Step 4: Learn CPTs from the observed data using Maximum Likelihood Estimation
def learn_cpts(model, df):

    # Learn CPTs using Maximum Likelihood Estimation
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

model = learn_cpts(model, df_reorganized)

# Step 5: Perform inference to predict missing values for a household
def perform_inference(model, df, household_node):
    inference = VariableElimination(model)

    # Get neighbors for the specified household
    neighbors_list = neighbors[household_node]

    # Select evidence based on neighbors' 'House' values, using the proper column names
    evidence = {}
    for neighbor in neighbors_list:
        neighbor_house_column = f'{neighbor}_House'  # Adjust the column name to match the reorganized DataFrame
        evidence[neighbor_house_column] = df.loc[0, neighbor_house_column]  # Assuming df has a single row (or adjust accordingly)

    # Predict missing values for the 'House' attribute of the specified household
    household_house_column = f'{household_node}_House'
    result = inference.query(variables=[household_house_column], evidence=evidence)

    print(result)


# Perform inference for a specific household node, e.g., 'Household 1'
perform_inference(model, df_reorganized, household_node='Household 1')

# Print the final dataframe
print(df_reorganized)
