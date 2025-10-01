import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

csv_path = os.path.join(os.path.dirname(__file__), "trades.csv")
df = pd.read_csv(csv_path)

df = df[df["profit"].notnull() & (df["profit"] != 0)]

print("✅ Dataset carregado com sucesso!")
print("Formato:", df.shape)
df.head()

df["success"] = (df["profit"] > 10).astype(int)
print("✅ Variável 'success' criada!")

# Distribuição inicial
print(df["success"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x=df["success"], palette="viridis")
plt.title("Distribuição de Trades (Sucesso x Falha)")
plt.xlabel("success")
plt.ylabel("Quantidade")
plt.show()

X = df[["ativo", "name", "tipo", "timeframe", "setup",
        "nivelDeEntrada", "stopLoss", "nivelDeAlvo",
        "riscoLoss", "riscoProfit"]].copy()
y = df["success"]

X["risk_reward_ratio"] = X["riscoProfit"] / (X["riscoLoss"] + 1e-6)
X["alvo_distancia"]   = X["nivelDeAlvo"] - X["nivelDeEntrada"]
X["stop_distancia"]   = X["nivelDeEntrada"] - X["stopLoss"]
X["alvo_stop_ratio"]  = (X["nivelDeAlvo"] - X["nivelDeEntrada"]) / (X["nivelDeEntrada"] - X["stopLoss"] + 1e-6)
X['entrada_stop_diff'] = X['nivelDeEntrada'] - X['stopLoss']
X['entrada_alvo_diff'] = X['nivelDeAlvo'] - X['nivelDeEntrada']
X['alvo_stop_diff'] = X['nivelDeAlvo'] - X['stopLoss']
X['abs_profit_loss_ratio'] = abs(X['riscoProfit']) / (abs(X['riscoLoss']) + 1e-6)
X['distancia_total'] = abs(X['alvo_distancia']) + abs(X['stop_distancia'])
X["stop_pct"] = (X["nivelDeEntrada"] - X["stopLoss"]) / X["nivelDeEntrada"]
X["alvo_pct"] = (X["nivelDeAlvo"] - X["nivelDeEntrada"]) / X["nivelDeEntrada"]
df["createdAt"] = pd.to_datetime(df["createdAt"])
X["dayofweek"] = df["createdAt"].dt.dayofweek
X["hour"] = df["createdAt"].dt.hour
X["is_weekend"] = df["createdAt"].dt.dayofweek >= 5
X["is_morning"] = df["createdAt"].dt.hour.between(6, 12).astype(int)
X["range_trade"] = (df["nivelDeAlvo"] - df["stopLoss"]) / df["nivelDeEntrada"]
X["spread_stop_alvo"] = (df["nivelDeAlvo"] - df["stopLoss"])
X["spread_stop_entrada"] = (df["nivelDeEntrada"] - df["stopLoss"])
X["spread_alvo_entrada"] = (df["nivelDeAlvo"] - df["nivelDeEntrada"])

label_encoders = {}
categorical_columns = ["ativo", "name", "tipo", "timeframe", "setup"]

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X.head()

num_cols = [
    "nivelDeEntrada", "stopLoss", "nivelDeAlvo",
    "riscoLoss", "riscoProfit",
    "risk_reward_ratio", "alvo_distancia",
    "stop_distancia", "alvo_stop_ratio",
    "entrada_stop_diff", "entrada_alvo_diff",
    "alvo_stop_diff", "abs_profit_loss_ratio",
    "distancia_total", "stop_pct", "alvo_pct",
    "dayofweek", "hour", "is_weekend", "is_morning",
    "range_trade", "spread_stop_alvo",
    "spread_stop_entrada", "spread_alvo_entrada"
]

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print("✅ Variáveis numéricas escalonadas com StandardScaler!")
X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Split realizado!")
print("Treino:", X_train.shape, " | Teste:", X_test.shape)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("✅ Dados balanceados com SMOTE!")
print("Antes:", y_train.value_counts().to_dict())
print("Depois:", y_train_res.value_counts().to_dict())

plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res, palette="coolwarm")
plt.title("Distribuição após SMOTE")
plt.xlabel("success")
plt.ylabel("Quantidade")
plt.show()

param_grid = {
    "n_estimators": [500],
    "max_depth": [50],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["sqrt"],
    "criterion": ["entropy"],
    "bootstrap": [False],
}

grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    n_iter=10
)

print("🚀 Iniciando treinamento...")
grid.fit(X_train_res, y_train_res)
best_model = grid.best_estimator_

print("✅ Treinamento concluído!")
print("Melhores hiperparâmetros:", grid.best_params_)

y_pred = best_model.predict(X_test)
print("=== Avaliação RandomForest otimizado ===")
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Falha","Sucesso"],
            yticklabels=["Falha","Sucesso"])
plt.title("Matriz de Confusão")
plt.ylabel("Real")
plt.xlabel("Previsto")
plt.show()

importances = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features, palette="mako")
plt.title("Importância das Features - RandomForest")
plt.show()

joblib.dump(best_model, "modelo_trades.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Modelo, encoders e scaler salvos em disco!")

"""# Nova seção"""