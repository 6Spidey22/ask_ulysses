from flask import Flask, render_template, request, jsonify, session
import ollama

app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    return render_template("ask_ulysses.html")

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)
