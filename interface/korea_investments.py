import os
import requests as req
from urllib.parse import urljoin
from typing import Dict, Any

class KoreaInvestments:
    def __init__(self, agent):
        self.agent = agent
        
        # if agent == "real":
        self.base_url = "https://openapi.koreainvestment.com:9443"
        # elif agent == "simulation":
        #     self.base_url = "https://openapi.koreainvestment.com:29443"
            
        self.default_headers = {
            'content-type': 'application/json; charset=UTF-8'
        }
        self.default_params = {}
        
        self.ws_session = None
        
    def set_default_headers(self, headers: Dict[str, str]):
        self.default_headers.update(headers)
        
    def set_default_params(self, params: Dict[str, Any]):
        self.default_params.update(params)
        
    def get(self, endpoint: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None):
        try:
            url = urljoin(self.base_url, endpoint)
            headers = {**self.default_headers, **(headers or {})}
            params = {**self.default_params, **(params or {})}
            res = req.get(url, headers=headers, params=params)
            
            return res
            
        except req.RequestException as e:
            return {"error": str(e)}
        
        finally:
            self.default_headers = {
                'Content-Type': 'application/json; charset=UTF-8'
            }
            
            self.default_params = {}
    
    def post(self, endpoint: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None):
        try:
            url = urljoin(self.base_url, endpoint)
            headers = {**self.default_headers, **(headers or {})}
            res = req.post(url, headers=headers, json=data)
            return res
        
        except req.RequestException as e:
            return {"error": str(e)}
        
        finally:
            self.default_headers = {
                'Content-Type': 'application/json; charset=UTF-8'
            }
            
            self.default_params = {}