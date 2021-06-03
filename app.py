

import numpy as np
# render_template redirect to the home page in index.html
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)  # to initialize the flask

model = pickle.load(open('model.pkl', 'rb'))

# define from where the user inout is getting


@app.route('/')
def home():
    return render_template('web.html')

# the user input is fed to the model.py to get the predicted value and return the result


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    out=prediction[0]
    if out == True:
        pr = "YES"
    else:
        pr="NO"

    # display the result in same html page
    return render_template('web.html', prediction_text='Diabate status = {}'.format(pr))



if __name__ == "__main__":
    app.run(debug=True)
