import logging
import time
import json
import os
from mt5 import TradeManager
from trade_ideas import get_trades_ideas

HISTORICO_JSON = "trades_processados.json"

def carregar_historico():
    """Carrega trades já processados do JSON"""
    if os.path.exists(HISTORICO_JSON):
        with open(HISTORICO_JSON, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def salvar_historico(trades_ids):
    """Salva trades processados no JSON"""
    with open(HISTORICO_JSON, "w", encoding="utf-8") as f:
        json.dump(list(trades_ids), f, ensure_ascii=False, indent=2)

def main():
    # Configuração de logs
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Main")

    # Carregar histórico
    trades_processados = carregar_historico()

    # Inicializar conexão MT5
    tm = TradeManager(
        logger=logger,
        mt5_path="C:/Program Files/MetaTrader 5 - IA/terminal64.exe",
        login=61409959,
        password="JoaoSolis1234!",
        server="Pepperstone-Demo",
        threshold=0.5
    )

    if not tm.connected:
        logger.error("❌ Não foi possível conectar ao MT5")
        return

    while True:
        logger.info("🔄 Buscando novos trades...")

        # Buscar trades do TradeIdeas
        trades = get_trades_ideas()

        if not trades:
            logger.info("⚠️ Nenhum trade disponível.")
        else:
            for trade in trades:
                trade_id = trade["Id"]

                # Se já processado, pular
                if trade_id in trades_processados:
                    continue

                logger.info(f"📌 Avaliando trade: {trade['Ativo']} ({trade['Tipo']}) | Setup: {trade['Setup']}")

                resultado = tm.send_order(trade)

                if resultado and resultado.get("success"):
                    logger.info(
                        f"✅ Trade EXECUTADO: {trade['Ativo']} | Probabilidade {resultado['prob_sucesso']*100:.1f}%"
                    )

                # Adicionar ao histórico mesmo se rejeitado
                trades_processados.add(trade_id)
                salvar_historico(trades_processados)

        logger.info("⏳ Aguardando 5 minutos antes da próxima execução...")
        time.sleep(300)  # 300s = 5min

    # Encerrar conexão
    tm.shutdown()

if __name__ == "__main__":
    main()
