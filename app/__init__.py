from flask import Flask, render_template, request
import joblib 
import numpy as np
import pandas as pd

def app_factory():
    app = Flask(__name__)

    model = joblib.load('project3_model_.pkl')
    @app.route("/", methods=['GET', 'POST'])
    def index():
        if request.method == 'GET':
            return render_template('index.html')
        if request.method == 'POST':
            WHO_region = request.form.get('WHO_region', False)
            New_cases = int(request.form['New_cases'])
            Cumulative_cases =int(request.form['Cumulative_cases'])
            Cumulative_deaths = int(request.form['Cumulative_deaths'])
            year = int(request.form['year'])
            month = int(request.form['month'])
            day = int(request.form['day'])
            dow = int(request.form['dow'])
            woy = int(request.form['woy'])
        entrance_subtotal = 0
        data = np.array([WHO_region, New_cases, Cumulative_cases, Cumulative_deaths, year, month, day, dow, woy]).reshape(1, -1)
        data = pd.DataFrame(data)
        
        entrance_subtotal = model.predict(data)

        return render_template('index.html', entrance_subtotal=entrance_subtotal)
    return app
if __name__ == '__main__':
    app = app_factory()
    app.run(debug=True)