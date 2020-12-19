from flask import Flask, render_template, request
from src.utils import *

import os
import random


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/start', methods=["GET"])
def insert():

    if request.values:
        print(request.values["choose"])

        # Charger une image aléatoire
        img_list = os.listdir("static/camera")
        img = random.choice(img_list)

        return render_template('insert.html', data=img)
    
    else :
        return render_template('insert.html', data="")


@app.route('/waste/pick-type')
def pick_type():
    close_waste_slot()

    return render_template('type.html')


@app.route('/confirmation', methods=['POST'])
def confirmation():
    waste_type = request.form['type']

    process_waste(waste_type)
    return render_template('confirmation.html')



if __name__ == "__main__":
    app.run(debug=True)
