import time
import hashlib
import requests
import concurrent.futures
from typing import Dict, Any

def fetch_page(url: str, page_offset: int = None) -> Dict[str, Any]:
    if page_offset is not None:
        url = f"{url}&page_offset={page_offset}"
    response = requests.get(url)
    return response.json()

def get_trades_ideas():
    expiry = int(time.time()) + (3 * 24 * 60 * 60)
    
    userid = "BlackBots"  
    account_type = "0"    
    secret_key = "72b7-543ba0a"  

    token_string = f"{userid}|{account_type}|{expiry}{secret_key}"
    
    token = hashlib.md5(token_string.encode()).hexdigest()
    
    base_url = f"https://component.autochartist.com/to/resources/results?account_type=LIVE&broker_id=958&token={token}&expire={expiry}&user=BlackBots&locale=pt-BR"
    
    first_page_data = fetch_page(base_url)
    
    page_info = first_page_data.get('page', {})
    total_pages = page_info.get('total_pages', 1)
    page_limit = page_info.get('limit', 20)
    
    all_trades = []
    all_trades.extend(first_page_data['items'])
    
    if total_pages > 1:
        page_offsets = [page * page_limit for page in range(1, total_pages)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, total_pages-1)) as executor:
            future_to_page = {
                executor.submit(fetch_page, base_url, offset): offset 
                for offset in page_offsets
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                offset = future_to_page[future]
                try:
                    page_data = future.result()
                    all_trades.extend(page_data['items'])
                except Exception:
                    pass
    
    trades = all_trades
    
    analises_tecnicas = []

    for trade in trades:
        chart_image = next((link['href'] for link in trade['links'] if link['rel'] == 'chart-xs'), None)
        
        analise = {
            "Id": str(trade['data']['result_uid']),
            "Ativo": trade['data']['symbol'],
            "Name": trade['data']['symbol_name'],
            "Tipo":'compra' if trade['data']['direction'] == 1 else 'venda',
            "Timeframe": trade['data']['interval'],
            "Setup": trade['data']['pattern'],
            "Identificadoem": trade['data']['identified'],
            "Análise": trade['data']['analysis_text'],
            "NíveldeEntrada": trade['data']['signal_levels']['entry_level'],
            "StopLoss": trade['data']['signal_levels']['stop_loss'],
            "filePath": chart_image,
            "NíveldeAlvo": trade['data']['signal_levels']['target_level'],
            "PeríododeAlvo": trade['data']['signal_levels']['target_period'],
            "Tipo de Mercado": "Internacional",
            "UserId": "BlackBots - Trade Ideas",
        }
        analises_tecnicas.append(analise)

    return analises_tecnicas

class TradeIdeas:
    """Classe para gerenciar trades do sistema TradeIdeas"""
    
    def __init__(self):
        pass
    
    def get_new_trades(self):
        """Retorna novos trades do sistema"""
        return get_trades_ideas()