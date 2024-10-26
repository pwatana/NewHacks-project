from flask import Flask, render_template, flash, request, session


app = Flask(__name__)

@app.route('/')
def main(): 
    return "Hello World"

@app.route('/form')
def form(): 
    """ Report the live condition of disasters """


if __name__ == "__main__": 
    app.run(debug=True)

