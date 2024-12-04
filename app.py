from flask import Flask, render_template, request, jsonify
from chat import get_response, generate_response
from langdetect import detect


app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    language = "thai" if detect(text) == 'th' else "english"
    # check: if text is valid    
    response = get_response(text, language)
    print(response)
    if response is None or response.lower().strip() == "i do not understand...":
        ## Typhoon 8b instruct model
        typhoon_response = generate_response(text)
        t_message = {"answer": typhoon_response}
        return jsonify(t_message)
    else:
        ## NLP model
        message = {"answer": response}    
        return jsonify(message)
    

if __name__ == "__main__":
    app.run(port=80, debug=True, use_reloader=False)


## uploaded file excel
## insert sqlite for checking data
## checking data in sqlite for find a same data; checking a format
## 