from flask import Flask, render_template, request, session
from ask_ulysses import tool_llm_response
import markdown
from bs4 import BeautifulSoup

app = Flask(__name__, template_folder='templates')

@app.route("/", methods=["GET", "POST"])
def index():
    response_html = None #default

    if request.method == "POST":
        user_question = request.form.get("question")
        response = tool_llm_response(user_question)
        response = response["message"]["content"]
        print(response)
        response_html = markdown.markdown(response)
        soup = BeautifulSoup(response_html, "html.parser")
        for a in soup.find_all("a"):
            a["target"] = "_blank"
            a["rel"] = "noopener noreferrer"

        response_html = str(soup)

    return render_template("ask_ulysses.html", response=response_html)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)
