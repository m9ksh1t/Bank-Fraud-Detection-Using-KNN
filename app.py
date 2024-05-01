from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb
import pickle

app = Flask(__name__)

# Load the pickled model
with open('pkl_models/ABC_model.pkl', 'rb') as model_file_ABC:
    modelABC = pickle.load(model_file_ABC)
with open('pkl_models/DTC_model.pkl', 'rb') as model_file_DTC:
    modelDTC = pickle.load(model_file_DTC)
with open('pkl_models/KNN_model.pkl', 'rb') as model_file_KNN:
    modelKNN = pickle.load(model_file_KNN)
with open('pkl_models/RFC_model.pkl', 'rb') as model_file_KNN:
    modelRFC = pickle.load(model_file_KNN)
with open('pkl_models/XGB_model.pkl', 'rb') as model_file_XGB:
    modelXGB = pickle.load(model_file_XGB)

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/predict_fraud', methods=['POST'])
def predict_fraud(): 
        if request.method =='POST':
            type = float(request.form.get('type'))
            amount = float(request.form.get('amount'))
            oldbalanceOrg = float(request.form.get('oldbalanceOrg'))
            newbalanceOrig = float(request.form.get('newbalanceOrig'))
            oldbalanceDest = int(request.form.get('oldbalanceDest'))
            newbalanceDest = int(request.form.get('newbalanceDest'))
                        
        user_input = {
            'type': type,  # transaction type
            'amount': amount,  # amount of the transaction
            'oldbalanceOrg': oldbalanceOrg,  # original balance of sender before the transaction
            'newbalanceOrig': newbalanceOrig,  # new balance of sender after the transaction
            'oldbalanceDest': oldbalanceDest,  # original balance of receiver before the transaction
            'newbalanceDest': newbalanceDest  # new balance of receiver after the transaction
        }
        
        
        new_data_df = pd.DataFrame([user_input])
        new_data_df.values
        
        dtest = xgb.DMatrix(new_data_df)
        
        predictionABC = modelABC.predict([[type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest]])
        predictionDTC = modelDTC.predict([[type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest]])
        predictionKNN = modelKNN.predict([[type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest]])
        predictionRFC = modelRFC.predict([[type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest]])
        predictionXGB = modelXGB.predict(dtest)
        
        pred_list = [predictionABC[0], predictionDTC[0], predictionKNN[0], predictionRFC[0], predictionXGB[0]]
        
        count_0 = pred_list.count(0)
        count_1 = pred_list.count(1)
        
        is_fraud = str(count_0 < count_1)
        
        return render_template('predict_fraud.html', result=is_fraud)

if __name__ == '__main__':
    app.run(debug=True)