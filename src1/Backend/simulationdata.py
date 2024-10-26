import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of households
num_households = 5

# Create household names
household_names = [f'Household {i+1}' for i in range(num_households)]

# Generate random "Yes" or "No" data for each category
data = {
    'Food': np.random.choice(['Yes', 'No'], num_households),
    'Water': np.random.choice(['Yes', 'No'], num_households),
    'Electricity': np.random.choice(['Yes', 'No'], num_households),
    'House': np.random.choice(['Yes', 'No'], num_households),
    'Temporary Shelter': np.random.choice(['Yes', 'No'], num_households),
    'Injury': np.random.choice(['Yes', 'No'], num_households)
}

# Create DataFrame
df_households = pd.DataFrame(data, index=household_names)

# Save to CSV
df_households.to_csv('household_data_yes_no.csv')

# Display DataFrame
print(df_households)
