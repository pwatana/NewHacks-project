from flask import Flask
app = Flask(__name__)

@app.route('/')
def main(): 
    return "Hello World"

@app.route('/form')
def form(): 
    return 

if __name__ == "__main__": 
    app.run(debug=True)

