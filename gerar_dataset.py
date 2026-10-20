import random
import pandas as pd

positivas = [
    "Eu adorei este produto, é excelente!",
    "O filme foi incrível, quero assistir de novo.",
    "Estou muito satisfeito com o serviço.",
    "O atendimento foi rápido e eficiente.",
    "O produto superou todas as minhas expectativas.",
    "Estou muito feliz com a compra.",
    "O atendimento foi excelente, parabéns!",
    "A experiência foi maravilhosa.",
    "Adorei o design e a qualidade.",
    "O serviço é de primeira!"
]

negativas = [
    "Odeio esperar tanto tempo, péssimo serviço.",
    "Estou muito insatisfeito com a qualidade.",
    "O filme foi horrível e muito chato.",
    "Não gostei do produto, dinheiro perdido.",
    "Experiência terrível, não recomendo.",
    "O atendimento foi lento e desrespeitoso.",
    "Estou muito decepcionado com a compra.",
    "A qualidade é péssima, não vale a pena.",
    "Nunca mais quero comprar aqui.",
    "O serviço foi horrível e demorado."
]

# Gerar 500 positivas e 500 negativas
frases = []
for _ in range(500):
    frases.append((random.choice(positivas), "positivo"))
    frases.append((random.choice(negativas), "negativo"))

# Embaralhar
random.shuffle(frases)

# Salvar em CSV
df = pd.DataFrame(frases, columns=["texto", "sentimento"])
df.to_csv("sentimentos.csv", index=False, encoding="utf-8")

print("Dataset gerado com sucesso! Total de linhas:", len(df))
