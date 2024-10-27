from flask import Flask, render_template, request, jsonify
import csv
import os

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

        numberinput = request.form['numberinput']
        house = request.form['house']

        fieldnames = ["name", "contactInfo", "disasterType", "severity", "hazards", "casualties", "shelter", "food", "water", "electricity", "numberinput", "house", "location"]

        with open('../data/data.csv','a') as inFile:
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            # writerow() will write a row in your csv file
            writer.writerow({"name": name, "contactInfo": contactInfo,
                            "disasterType": disasterType, "severity": severity, 
                            "hazards": hazards, "casualties": casualties, 
                            "shelter": shelter, "food": food, "water": water, 
                            "electricity": electricity, "numberinput":numberinput, "house": house,
                            "location": (latitude, longitude)})
    return render_template('form.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"}), 400

    image = request.files['image']
    numbers = []
    for filename in os.listdir("../data/images"):
        numbers.append(int(filename.split('.')[0]))
    next_number = max(numbers) + 1 if numbers else 1
    image.save("../data/images/" + str(next_number) + ".png")  # Save the image to the photos folder
    print("Saved!")
    return jsonify({"success": True, "message": "Image saved successfully!"})

if __name__ == "__main__": 
    app.run(debug=True)

