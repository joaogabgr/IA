# ===============================================
# avaliar_trade.py
# Carrega modelo salvo e avalia trades novos
# ===============================================

import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def carregar_modelo():
    modelo_path = os.path.join(BASE_DIR, "modelo_trades.pkl")
    encoders_path = os.path.join(BASE_DIR, "encoders.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    modelo = joblib.load(modelo_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    return modelo, encoders, scaler

def avaliar_trade(trade_dict, modelo, encoders, scaler, threshold=0.5):
    # Criar DataFrame com o trade
    novo_trade = pd.DataFrame([trade_dict])
    
    # Verificar se createdAt está presente, se não, usar data atual
    if "createdAt" not in novo_trade.columns:
        novo_trade["createdAt"] = pd.Timestamp.now()
    
    # Converter createdAt para datetime
    novo_trade["createdAt"] = pd.to_datetime(novo_trade["createdAt"])

    # ===== CRIAR TODAS AS FEATURES EXTRAS =====
    
    # Features básicas de risco/recompensa
    novo_trade["risk_reward_ratio"] = novo_trade["riscoProfit"] / (novo_trade["riscoLoss"] + 1e-6)
    novo_trade["alvo_distancia"] = novo_trade["nivelDeAlvo"] - novo_trade["nivelDeEntrada"]
    novo_trade["stop_distancia"] = novo_trade["nivelDeEntrada"] - novo_trade["stopLoss"]
    novo_trade["alvo_stop_ratio"] = (novo_trade["nivelDeAlvo"] - novo_trade["nivelDeEntrada"]) / (novo_trade["nivelDeEntrada"] - novo_trade["stopLoss"] + 1e-6)
    
    # Features de diferenças
    novo_trade['entrada_stop_diff'] = novo_trade['nivelDeEntrada'] - novo_trade['stopLoss']
    novo_trade['entrada_alvo_diff'] = novo_trade['nivelDeAlvo'] - novo_trade['nivelDeEntrada']
    novo_trade['alvo_stop_diff'] = novo_trade['nivelDeAlvo'] - novo_trade['stopLoss']
    novo_trade['abs_profit_loss_ratio'] = abs(novo_trade['riscoProfit']) / (abs(novo_trade['riscoLoss']) + 1e-6)
    novo_trade['distancia_total'] = abs(novo_trade['alvo_distancia']) + abs(novo_trade['stop_distancia'])
    
    # Features percentuais
    novo_trade["stop_pct"] = (novo_trade["nivelDeEntrada"] - novo_trade["stopLoss"]) / novo_trade["nivelDeEntrada"]
    novo_trade["alvo_pct"] = (novo_trade["nivelDeAlvo"] - novo_trade["nivelDeEntrada"]) / novo_trade["nivelDeEntrada"]
    
    # Features temporais
    novo_trade["dayofweek"] = novo_trade["createdAt"].dt.dayofweek
    novo_trade["hour"] = novo_trade["createdAt"].dt.hour
    novo_trade["is_weekend"] = novo_trade["createdAt"].dt.dayofweek >= 5
    novo_trade["is_morning"] = novo_trade["createdAt"].dt.hour.between(6, 12).astype(int)
    
    # Features de spread e range
    novo_trade["range_trade"] = (novo_trade["nivelDeAlvo"] - novo_trade["stopLoss"]) / novo_trade["nivelDeEntrada"]
    novo_trade["spread_stop_alvo"] = (novo_trade["nivelDeAlvo"] - novo_trade["stopLoss"])
    novo_trade["spread_stop_entrada"] = (novo_trade["nivelDeEntrada"] - novo_trade["stopLoss"])
    novo_trade["spread_alvo_entrada"] = (novo_trade["nivelDeAlvo"] - novo_trade["nivelDeEntrada"])

    # ===== APLICAR ENCODERS PARA VARIÁVEIS CATEGÓRICAS =====
    categorical_columns = ["ativo", "name", "tipo", "timeframe", "setup"]
    for col in categorical_columns:
        if col in novo_trade.columns:
            try:
                novo_trade[col] = encoders[col].transform(novo_trade[col])
            except ValueError as e:
                print(f"⚠️ Aviso: Valor desconhecido para '{col}': {novo_trade[col].iloc[0]}")
                # Em caso de valor desconhecido, usar o valor mais comum (0)
                novo_trade[col] = 0

    # ===== APLICAR STANDARDSCALER PARA VARIÁVEIS NUMÉRICAS =====
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
    
    # Aplicar scaler apenas nas colunas numéricas
    novo_trade[num_cols] = scaler.transform(novo_trade[num_cols])

    # ===== REMOVER COLUNA CREATEDAT (USADA APENAS PARA CRIAR FEATURES TEMPORAIS) =====
    if 'createdAt' in novo_trade.columns:
        novo_trade = novo_trade.drop('createdAt', axis=1)

    # ===== GARANTIR QUE TODAS AS COLUNAS SEJAM NUMÉRICAS =====
    novo_trade = novo_trade.apply(pd.to_numeric, errors="coerce")

    # ===== FAZER PREDIÇÃO =====
    try:
        prob = modelo.predict_proba(novo_trade)[0]
        print(f"📊 Probabilidade de dar certo: {prob[1]*100:.1f}%")
        return prob[1]
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        print(f"Shape do input: {novo_trade.shape}")
        print(f"Colunas: {list(novo_trade.columns)}")
        return None

# Função auxiliar para testar o script
def testar_avaliacao():
    """Função para testar se o script está funcionando corretamente"""
    try:
        modelo, encoders, scaler = carregar_modelo()
        print("✅ Modelo, encoders e scaler carregados com sucesso!")
        
        # Exemplo de trade para teste
        trade_exemplo = {
            "ativo": "EURUSD",
            "name": "Trade Exemplo",
            "tipo": "BUY",
            "timeframe": "H1",
            "setup": "Breakout",
            "nivelDeEntrada": 1.1000,
            "stopLoss": 1.0950,
            "nivelDeAlvo": 1.1100,
            "riscoLoss": -50,
            "riscoProfit": 100,
            "createdAt": "2024-01-15 10:30:00"
        }
        
        resultado = avaliar_trade(trade_exemplo, modelo, encoders, scaler)
        if resultado is not None:
            print("✅ Teste realizado com sucesso!")
        else:
            print("❌ Erro no teste!")
            
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")

if __name__ == "__main__":
    testar_avaliacao()