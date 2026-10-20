import pandas as pd
import re
import nltk
import unicodedata
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar dataset
df = pd.read_csv("sentimentos.csv")
print("Primeiras linhas do dataset:")
print(df.head())
print("Total de linhas carregadas:", len(df))

# 2. Limpeza do texto
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("portuguese")

def limpar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)
    return texto

df["texto_limpo"] = df["texto"].apply(limpar_texto)

# 3. Vetorização
vectorizer = TfidfVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(df["texto_limpo"])
y = df["sentimento"]

# 4. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Treinar modelos
modelos = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

acuracias = {}
melhor_modelo = None
melhor_nome = ""
melhor_acc = 0

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acuracias[nome] = acc
    print(f"\n===== {nome} =====")
    print("Acurácia:", acc)
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))
    if acc > melhor_acc:
        melhor_acc = acc
        melhor_modelo = modelo
        melhor_nome = nome

# 6. Gráfico de barras das acurácias
plt.figure(figsize=(6, 4))
plt.bar(acuracias.keys(), acuracias.values(), color=["blue", "green", "red"])
plt.title("Comparação de Acurácia dos Modelos")
plt.ylabel("Acurácia")
plt.ylim(0, 1)
plt.savefig("comparacao_acuracia.png")
plt.close()

print("✅ Gráfico de comparação de acurácia salvo como 'comparacao_acuracia.png'")

# 7. Matriz de confusão do melhor modelo
y_pred = melhor_modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=["positivo", "negativo"])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["positivo", "negativo"],
            yticklabels=["positivo", "negativo"])
plt.title(f"Matriz de Confusão - {melhor_nome}")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig("matriz_confusao.png")
plt.close()

print(f"✅ Matriz de confusão salva como 'matriz_confusao.png' para o modelo {melhor_nome}")

# 8. Salvar modelo vencedor
with open("modelo_sentimentos.pkl", "wb") as f:
    pickle.dump((vectorizer, melhor_modelo), f)

print("\nModelo treinado e salvo como 'modelo_sentimentos.pkl'")
