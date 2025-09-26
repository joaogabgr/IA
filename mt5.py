import MetaTrader5 as mt5
from IA.avaliar_trades import carregar_modelo, avaliar_trade

class TradeManager:
    def __init__(self, logger, mt5_path=None, login=None, password=None, server=None, threshold=0.5):
        self.logger = logger
        self.threshold = threshold

        if mt5_path:
            if not mt5.initialize(path=mt5_path):
                self.logger.error(f"Erro ao inicializar MT5 no path: {mt5_path} | {mt5.last_error()}")
                self.connected = False
                return
        else:
            if not mt5.initialize():
                self.logger.error(f"Erro ao inicializar MT5 | {mt5.last_error()}")
                self.connected = False
                return

        self.connected = True

        if login and password and server:
            authorized = mt5.login(login=login, password=password, server=server)
            if not authorized:
                self.logger.error(f"Falha ao logar na conta {login} | {mt5.last_error()}")
                self.connected = False
                return
            else:
                self.logger.info(f"✅ Logado com sucesso na conta {login} no servidor {server}")

        # Carregar modelo IA
        self.modelo, self.encoders = carregar_modelo()

    # -------------------------------
    # Fechar conexão com MT5
    # -------------------------------
    def shutdown(self):
        mt5.shutdown()
        self.connected = False
        self.logger.info("🔌 Conexão com MT5 encerrada.")

    # -------------------------------
    # Utilitários
    # -------------------------------
    def get_symbol_info(self, symbol):
        if not self.connected:
            self.logger.error("MT5 não está conectado")
            return None
        info = mt5.symbol_info(symbol)
        if not info:
            self.logger.warning(f"Símbolo {symbol} não encontrado")
        return info

    def get_current_price(self, symbol, trade_type):
        """Retorna preço atual de compra/venda"""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            self.logger.error(f"Não foi possível obter tick do símbolo {symbol}")
            return None
        return tick.bid if trade_type.lower() == 'venda' else tick.ask

    # -------------------------------
    # Cálculo de risco genérico
    # -------------------------------
    def calculate_risk(self, symbol, trade_type, lot_size, entry_price, target_price):
        """Calcula risco (loss ou profit)"""
        if not self.connected or not target_price:
            return 0.0

        try:
            if lot_size <= 0 or entry_price <= 0 or target_price <= 0:
                return 0.0

            if trade_type.lower() == 'compra':
                order_type = mt5.ORDER_TYPE_BUY
            else:
                order_type = mt5.ORDER_TYPE_SELL

            profit = mt5.order_calc_profit(order_type, symbol, lot_size, entry_price, target_price)
            if profit is None:
                return 0.0

            # se target < entry → risco de perda, senão → risco de ganho
            return round(abs(profit), 2)
        except Exception as e:
            self.logger.error(f"Erro ao calcular risco {symbol}: {e}")
            return 0.0

    # -------------------------------
    # Cálculo de lote normalizado
    # -------------------------------
    def calculate_normalized_lot(self, symbol, trade_type, entry_price, stop_loss, take_profit=None, risco_alvo=100.0):
        if not self.connected:
            self.logger.error("MT5 não está conectado")
            return None

        if not stop_loss or stop_loss <= 0:
            return None

        try:
            order_type = mt5.ORDER_TYPE_BUY if trade_type.lower() == 'compra' else mt5.ORDER_TYPE_SELL
            if not mt5.symbol_select(symbol, True):
                return None

            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None

            min_lot, max_lot, lot_step = symbol_info.volume_min, symbol_info.volume_max, symbol_info.volume_step
            risco_min_volume = mt5.order_calc_profit(order_type, symbol, min_lot, entry_price, stop_loss)
            if not risco_min_volume:
                return None

            risco_min_volume = abs(risco_min_volume)
            lote_ideal = (min_lot * risco_alvo) / risco_min_volume
            lote_ajustado = round(lote_ideal / lot_step) * lot_step
            lote_final = max(min_lot, min(max_lot, lote_ajustado))

            risco_loss = self.calculate_risk(symbol, trade_type, lote_final, entry_price, stop_loss)
            risco_profit = self.calculate_risk(symbol, trade_type, lote_final, entry_price, take_profit) if take_profit else 0

            return {
                'lote': round(lote_final, 2),
                'riscoLoss': risco_loss,
                'riscoProfit': risco_profit,
                'risk_reward_ratio': round(risco_profit / risco_loss, 2) if risco_loss > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Erro no cálculo de lote: {e}")
            return None

    # -------------------------------
    # Envio de ordens
    # -------------------------------
    def send_order(self, trade_data):
        if not self.connected:
            self.logger.error("MT5 não está conectado")
            return None

        try:
            symbol = trade_data['Ativo']
            trade_type = trade_data['Tipo']
            entry_price = float(trade_data['NíveldeEntrada'])
            stop_loss = float(trade_data['StopLoss'])
            take_profit = float(trade_data['NíveldeAlvo']) if trade_data['NíveldeAlvo'] else None
            trade_id = trade_data['Id']

            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Falha ao selecionar {symbol}")
                return None

            # Cálculo do lote e riscos
            lot_result = self.calculate_normalized_lot(
                symbol, trade_type, entry_price, stop_loss, take_profit, risco_alvo=100.0
            )
            if not lot_result:
                return None

            risco_loss = lot_result['riscoLoss']
            risco_profit = lot_result['riscoProfit']

            trade_dict = {
                "ativo": symbol,
                "tipo": trade_type,
                "timeframe": trade_data.get("TimeFrame", "60"),
                "setup": trade_data.get("Setup", "Desconhecido"),
                "nivelDeEntrada": entry_price,
                "stopLoss": stop_loss,
                "nivelDeAlvo": take_profit if take_profit else 0.0,
                "riscoLoss": risco_loss,
                "riscoProfit": risco_profit
            }

            # IA valida probabilidade
            prob_sucesso = avaliar_trade(trade_dict, self.modelo, self.encoders, threshold=self.threshold)
            if prob_sucesso < self.threshold:
                return {"success": False, "reason": "IA rejeitou trade", "prob_sucesso": prob_sucesso}

            self.logger.info(
                f"IA aprovou trade {symbol} | Probabilidade {prob_sucesso*100:.1f}%"
            )

            # 🚀 Enviar ordens
            results = []

            # 1️⃣ Ordem a mercado
            market_comment = f"{trade_id}_market"
            market_result = self._send_market_order(symbol, trade_type, stop_loss, take_profit, market_comment, lot_result['lote'], prob_sucesso*100)
            if market_result:
                results.append(market_result)

            # 2️⃣ Ordem pendente
            pending_comment = f"{trade_id}_pending"
            pending_result = self._send_pending_order(symbol, trade_type, entry_price, stop_loss, take_profit, pending_comment, prob_sucesso*100)
            if pending_result:
                results.append(pending_result)

            return {
                'success': True if results else False,
                'orders': results,
                'market_order': market_result,
                'pending_order': pending_result,
                'prob_sucesso': prob_sucesso,
                'risco_loss': risco_loss,
                'risco_profit': risco_profit
            }

        except Exception as e:
            self.logger.error(f"Erro ao enviar ordem: {e}")
            return None

    def _send_market_order(self, symbol, trade_type, stop_loss, take_profit, comment, lot_size, prob_sucesso):
        """Envia ordem a mercado imediata"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.error(f"Não foi possível obter tick de {symbol}")
                return None

            # Preço correto conforme tipo
            price = tick.ask if trade_type.lower() == 'compra' else tick.bid
            if price <= 0:
                self.logger.error(f"Preço inválido para {symbol}: {price}")
                return None

            order_type = mt5.ORDER_TYPE_BUY if trade_type.lower() == 'compra' else mt5.ORDER_TYPE_SELL

            filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC]

            for mode in filling_modes:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": price,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "deviation": 20,
                    "magic": 99999,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mode,
                    "comment": prob_sucesso,
                }

                result = mt5.order_send(request)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"✅ Ordem a mercado executada: {symbol} | ticket={result.order}")
                    return {
                        'order': result.order,
                        'risco_loss': stop_loss,
                        'risco_profit': take_profit
                    }

            return None

        except Exception as e:
            self.logger.error(f"Erro em ordem a mercado {symbol}: {e}")
            return None


    def _send_pending_order(self, symbol, trade_type, entry_price, stop_loss, take_profit, comment, prob_sucesso):
        """Envia ordem pendente"""
        try:
            lot_result = self.calculate_normalized_lot(symbol, trade_type, entry_price, stop_loss, take_profit, risco_alvo=100.0)
            if not lot_result:
                return None

            lot_size = lot_result['lote']
            current_price = self.get_current_price(symbol, trade_type)
            if not current_price:
                return None

            # Tipo de ordem
            if trade_type.lower() == 'compra':
                order_type = mt5.ORDER_TYPE_BUY_STOP if entry_price > current_price else mt5.ORDER_TYPE_BUY_LIMIT
            else:
                order_type = mt5.ORDER_TYPE_SELL_STOP if entry_price < current_price else mt5.ORDER_TYPE_SELL_LIMIT

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 88888,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
                "comment": prob_sucesso,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return {'order': result.order, 'risco_loss': lot_result['riscoLoss'], 'risco_profit': lot_result['riscoProfit']}
            return None
        except Exception as e:
            self.logger.error(f"Erro em ordem pendente {symbol}: {e}")
            return None
