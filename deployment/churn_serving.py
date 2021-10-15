import pickle

from flask import Flask, request, jsonify


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('../churn-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('churn')


@app.route('/predict', methods=['POST'])  # A
def predict():

    # A Assign the /predict route to the predict function
    # B Get the content of the request in JSON
    # C Score the customer
    # D Prepare the response
    # E Convert the response to JSON

    customer = request.get_json()  # B

    prediction = predict_single(customer, dv, model)  # C
    churn = prediction >= 0.5  # D

    result = {  # D
        'churn_probability': float(prediction),  # D
        'churn': bool(churn),  # D
    }  # D

    return jsonify(result)  # E


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
