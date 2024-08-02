from flask import Flask, request, render_template
from model import predict, X

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = []
        for column in X.columns:
            input_data.append(request.form[column])
        prediction = predict(input_data)
        return render_template('index.html', prediction=prediction, input_data=X.columns)
    return render_template('index.html', prediction=None, input_data=X.columns)

if __name__ == '__main__':
    app.run(debug=True)
