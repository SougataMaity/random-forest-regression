from flask import Flask, render_template, request
import numpy as np
import pickle

filename = "rfmodel.sav"
load_model = pickle.load(open(filename, "rb"))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template ('home.html')
@app.route('/pred', methods = ['POST',"GET"])
def predict():
    if request.method == "POST":
        v1 = request.form.get("var1")
        v2 = request.form.get("var2")
        v3 = request.form.get("var3")
        v4 = request.form.get("var4")
        v5 = request.form.get("var5")
        v6 = request.form.get("var6")
        v7 = request.form.get("var7")
        v8 = request.form.get("var8")
        v9 = request.form.get("var9")
        v10 = request.form.get("var10")
        v11 = request.form.get("var11")
        v12 = request.form.get("var12")
        v13 = request.form.get("var12")
        
        input = np.array([v1, v2, v3, v4, v5, v6,v7, v8, v9, v10, v11, v12,v13])
        value = input.astype(np.float_)
        pred = load_model.predict([value])[0]

    return render_template('predict.html', pred = '{}'.format(pred))
    
@app.route('/plot')
def plot():
    return render_template ('plot.html')

if __name__ == '__main__':
    app.run(debug=True)