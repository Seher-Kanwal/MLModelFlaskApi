from flask import Flask, request, jsonify
import pickle
import torch

model = pickle.load(open('model.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
app = Flask(__name__)


# Creating Routes
@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    tense = request.form.get('query')
    print(tense)
    token = tokenizer.encode(tense, return_tensors='pt')
    output = model.generate(token, num_beams=10, max_length=100, early_stopping=True, no_repeat_ngram_size=2,)
    res = tokenizer.decode(output[0], skip_special_tokens=True)
    print(res)
    result = {'query': res}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
