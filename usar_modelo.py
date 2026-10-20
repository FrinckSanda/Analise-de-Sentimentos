import pickle

# Carregar modelo salvo
with open("modelo_sentimentos.pkl", "rb") as f:
    vectorizer, modelo = pickle.load(f)

frases_teste = [
    "Eu adorei a qualidade do produto!",
    "O serviço foi horrível e demorado.",
    "Estou muito feliz com a compra.",
    "Nunca mais quero comprar aqui.",
    "O atendimento foi excelente, parabéns!"
]

for frase in frases_teste:
    X = vectorizer.transform([frase])
    resultado = modelo.predict(X)[0]
    print(f"{frase} -> {resultado}")
