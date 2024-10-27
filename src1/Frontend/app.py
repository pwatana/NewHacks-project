from flask import Flask, render_template, request, jsonify
import csv
import os
import pandas as pd


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
    # Load the CSV data
    df = pd.read_csv('../data/data.csv')
    
    # Extract latitude and longitude from the location tuple string
    locations = []
    for _, row in df.iterrows():
        location = row['location'].strip("()")  # Remove parentheses
        latitude, longitude = map(float, location.split(", "))  # Split by comma and space, and convert to floats
        locations.append({
            "name": row['name'],
            "contactInfo":row['contactInfo'],
            "disasterType": row['disasterType'],
            "latitude": latitude,
            "longitude": longitude
        })

    # Load the CSV data
    df2 = pd.read_csv('../data/data2.csv')
    
    # Extract latitude and longitude from the location tuple string
    locations2 = []
    for _, row in df2.iterrows():
        location2 = row['location'].strip("()")  # Remove parentheses
        latitude2, longitude2 = map(float, location2.split(", "))  # Split by comma and space, and convert to floats
        locations2.append({
            "name": row['name'],
            "contactInfo":row['contactInfo'],
            "disasterType": row['disasterType'],
            "latitude": latitude2,
            "longitude": longitude2
        })

    return jsonify({"locations1": locations, "locations2": locations2})  # Send JSON data to the frontend

@app.route('/results', methods=['GET'])
def results():

    return render_template('results.html')



if __name__ == "__main__": 
    app.run(debug=True)

