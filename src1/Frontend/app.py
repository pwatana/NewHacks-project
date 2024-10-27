from flask import Flask, render_template, request, jsonify
import csv
import os
import pandas as pd
import BayesianNetwork2


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        name = request.form['reporterName']
        contactInfo = request.form['contactInfo']
        disasterType = request.form['disasterType']
        severity = request.form['severity']
        hazards = request.form.getlist('hazards')
        casualties = request.form['casualties']
        shelter = request.form['shelter']
        food = request.form['food']
        water = request.form['water']
        electricity = request.form['electricity']
        latitude = request.form['latitude']
        longitude = request.form['longitude']

        people = request.form['people']
        house = request.form['house']

        fieldnames = ["name", "contactInfo", "disasterType", "severity", "hazards", "casualties", "shelter", "food", "water", "electricity", "people", "house", "location"]

        with open('../data/data.csv','a') as inFile:
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            # writerow() will write a row in your csv file
            writer.writerow({"name": name, "contactInfo": contactInfo,
                            "disasterType": disasterType, "severity": severity,
                            "hazards": hazards, "casualties": casualties,
                            "shelter": shelter, "food": food, "water": water,
                            "electricity": electricity, "people":people, "house": house,
                            "location": (float(latitude), float(longitude))})
    return render_template('form.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"}), 400

    image = request.files['image']
    numbers = []
    for filename in os.listdir("../data/images"):
        print(filename[-3:])
        if filename.lower().endswith(".png"):  # Case-insensitive check for .png
            print("appending")
            numbers.append(int(filename.split('.')[0]))
    print(numbers)
    next_number = max(numbers) + 1 if numbers else 1
    image.save("../data/images/" + str(next_number) + ".png")  # Save the image to the photos folder
    print("Saved!")
    return jsonify({"success": True, "message": "Image saved successfully!"})

@app.route('/locations')
def locations():
    # # Load the CSV data
    # df = pd.read_csv('../data/data.csv')
    #
    # # Extract latitude and longitude from the location tuple string
    # locations = []
    # for _, row in df.iterrows():
    #     location = row['location'].strip("()")  # Remove parentheses
    #     latitude, longitude = map(float, location.split(", "))  # Split by comma and space, and convert to floats
    #     locations.append({
    #         "name": row['name'],
    #         "contactInfo":row['contactInfo'],
    #         "disasterType": row['disasterType'],
    #         "latitude": latitude,
    #         "longitude": longitude
    #     })
    #
    # # Load the CSV data
    # df2 = pd.read_csv('../data/data2.csv')
    #
    # # Extract latitude and longitude from the location tuple string
    # locations2 = []
    # for _, row in df2.iterrows():
    #     location2 = row['location'].strip("()")  # Remove parentheses
    #     latitude2, longitude2 = map(float, location2.split(", "))  # Split by comma and space, and convert to floats
    #     locations2.append({
    #         "name": row['name'],
    #         "contactInfo":row['contactInfo'],
    #         "disasterType": row['disasterType'],
    #         "latitude": latitude2,
    #         "longitude": longitude2
    #     })

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

    neighbors = BayesianNetwork2.find_neighbors(df_households, max_distance=2)
    model = BayesianNetwork2.setup_bayesian_network(neighbors)
    output = BayesianNetwork2.perform_inference(model, df_reorganized, household_node='Household 1')

    return output  # Send JSON data to the frontend

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/results', methods=['GET'])
def results():

    return render_template('results.html')



if __name__ == "__main__":
    app.run(debug=True)

