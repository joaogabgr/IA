# ===============================================
# treinar_modelo.py
# Treina e salva o modelo RandomForest para trades
# ===============================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def treinar_e_salvar_modelo(csv_path="trades.csv"):
    df = pd.read_csv(csv_path)

    # Criar variável alvo
    df["success"] = (df["profit"] > 0).astype(int)

    # Seleção de features
    X = df[["ativo", "tipo", "timeframe", "setup",
            "nivelDeEntrada", "stopLoss", "nivelDeAlvo",
            "riscoLoss", "riscoProfit"]].copy()
    y = df["success"]

    # Feature engineering
    X["risk_reward_ratio"] = X["riscoProfit"] / (X["riscoLoss"] + 1e-6)
    X["alvo_distancia"] = X["nivelDeAlvo"] - X["nivelDeEntrada"]

    # Label Encoding
    label_encoders = {}
    for col in ["ativo", "tipo", "timeframe", "setup"]:
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balanceamento com SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # GridSearch para otimização
    param_grid = {
        "n_estimators": [500],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_res, y_train_res)
    best_model = grid.best_estimator_

    # Avaliação
    y_pred = best_model.predict(X_test)
    print("=== Avaliação RandomForest otimizado ===")
    print(classification_report(y_test, y_pred))

    # Salvar modelo e encoders
    joblib.dump(best_model, "modelo_trades.pkl")
    joblib.dump(label_encoders, "encoders.pkl")
    print("✅ Modelo e encoders salvos em disco.")

if __name__ == "__main__":
    treinar_e_salvar_modelo("trades.csv")
