from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session

from ask_ulysses import tool_llm_response
import markdown
from bs4 import BeautifulSoup


app = Flask(__name__, template_folder='templates')
app.secret_key = "Cornell_College_Ask_Ulysses"  # required for sessions
app.config["SESSION_TYPE"] = "filesystem"  # store session server-side
Session(app)

@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_question = request.form.get("question")
        response = tool_llm_response(user_question)
        response = response["message"]["content"]
        response_html = markdown.markdown(response)
        soup = BeautifulSoup(response_html, "html.parser")
        for a in soup.find_all("a"):
            a["target"] = "_blank"
            a["rel"] = "noopener noreferrer"

        response_html = str(soup)

        session["chat_history"].append({"user": user_question, "bot": response_html})
        session.modified = True
        return redirect(url_for("home"))

    return render_template("ask_ulysses.html", chat_history=session.get("chat_history", []))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)
