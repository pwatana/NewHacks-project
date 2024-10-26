from flask import Flask, render_template, request
import csv

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/form', methods=['GET', 'POST'])
def form(): 
    if request.method == 'POST':
        name = request.form['reporterName']
        contactInfo = request.form['contactInfo']
        disasterType = request.form['disasterType']
        hazards = request.form['hazards']
        casualties = request.form['casualties']
        propertyDamage = request.form['propertyDamage']
        shelter = request.form['shelter']
        food = request.form['food']
        water = request.form['water']
        electricity = request.form['electricity']

        #comment = request.form['comment']

        fieldnames = ["name", "contactInfo", "disasterType", "hazards", "casualties", "propertyDamange", "shelter", "food", "water"]

        with open('data.csv','w') as inFile:
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            # writerow() will write a row in your csv file
            writer.writerow({"name": name, "contactInfo": contactInfo, "disasterType": disasterType, "hazards": hazards, "casualties": casualties, "propertyDamange": propertyDamage, "shelter": shelter, "food": food, "water": water})
    return render_template('form.html')

if __name__ == "__main__": 
    app.run(debug=True)

