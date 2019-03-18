from flask import Flask, jsonify
from framework import response
from chatterbot import ChatBot

app = Flask(__name__)


@app.route("/api/chatbot/basic/<string:sentence>")
def basic(sentence):
    return  jsonify({"response":response(sentence)}) 



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')