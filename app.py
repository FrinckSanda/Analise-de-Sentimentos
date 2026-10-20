from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Carregar modelo
with open("modelo_sentimentos.pkl", "rb") as f:
    vectorizer, modelo = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    sentimento = None
    frase = None
    if request.method == "POST":
        frase = request.form["frase"]
        X = vectorizer.transform([frase])
        sentimento = modelo.predict(X)[0]
    return render_template("index.html", sentimento=sentimento, frase=frase)

if __name__ == "__main__":
    app.run(debug=True)
