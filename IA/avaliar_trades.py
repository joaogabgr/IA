# ===============================================
# avaliar_trade.py
# Carrega modelo salvo e avalia trades novos
# ===============================================

import pandas as pd
import joblib
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def carregar_modelo():
    modelo_path = os.path.join(BASE_DIR, "modelo_trades.pkl")
    encoders_path = os.path.join(BASE_DIR, "encoders.pkl")

    modelo = joblib.load(modelo_path)
    encoders = joblib.load(encoders_path)
    return modelo, encoders

def avaliar_trade(trade_dict, modelo, encoders, threshold=0.5):
    novo_trade = pd.DataFrame([trade_dict])

    # Features extras
    novo_trade["risk_reward_ratio"] = novo_trade["riscoProfit"] / (novo_trade["riscoLoss"] + 1e-6)
    novo_trade["alvo_distancia"] = novo_trade["nivelDeAlvo"] - novo_trade["nivelDeEntrada"]

    # Aplicar encoders
    for col in ["ativo", "name", "tipo", "timeframe", "setup"]:
        novo_trade[col] = encoders[col].transform(novo_trade[col])

    novo_trade = novo_trade.apply(pd.to_numeric, errors="coerce")

    prob = modelo.predict_proba(novo_trade)[0]

    print(f"📊 Probabilidade de dar certo: {prob[1]*100:.1f}%")

    return prob[1]