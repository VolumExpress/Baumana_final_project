from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf


app = Flask(__name__)


def get_prediction_depth(df):
    model = tf.keras.models.load_model(r"models\NN_model_for_depth")
    y_pred = model.predict(df)

    return f"Глубина сварного шва равна {y_pred}"

def get_prediction_width_width(df):
    model = tf.keras.models.load_model(r"models\NN_model_for_width")
    y_pred = model.predict(df)

    return f"Ширина сварного шва равна {y_pred}"

@app.route('/', methods=['post', 'get'])
def processing():
    message_d = ''
    message_w = ''
    if request.method == 'POST':
        IW = int(request.form.get('IW'))
        IF = int(request.form.get('IF'))
        VW = int(request.form.get('VW'))
        FP = int(request.form.get('FP'))
        
        stdIW = 1.6666666666666663
        meanIW = 45.666666666666664
        
        stdIF = 5.109903238918627
        meanIF = 141.33333333333334
        
        stdVW = 2.0467152244209603
        meanVW = 8.63888888888889
        
        stdFP = 21.343747458109483
        meanFP = 78.33333333333333      
        
        IW_n = (IW - meanIW) / stdIW
        IF_n = (IF - meanIF) / stdIF
        VW_n = (VW - meanVW) / stdVW
        FP_n = (FP - meanFP) / stdFP
        
        data = {'IW': [IW_n], 'IF': [IF_n], 'VW': [VW_n], 'FP': [FP_n]}

        df = pd.DataFrame(data=data)
        
        
        message_d = get_prediction_depth(df)
        message_w = get_prediction(df)

    return render_template('login.html', message_d=message_d, message_w=message_w)


if __name__ == '__main__':
    app.run()