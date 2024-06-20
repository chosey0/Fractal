
import requests
from interface.korea_investments import KoreaInvestments
from datetime import datetime, time, timedelta
import yaml

class Agent(KoreaInvestments):
    def __init__(self, name, config_path):
        super().__init__(name)
        self.name = name
        self._config_path = config_path
        
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
        if name not in self._config.keys():
            self.name = input("Re. Agent Name: ")
        
        self._app = self._config[self.name]["APP_KEY"]
        self._secret = self._config[self.name]["SECRET_KEY"]
        self._approval = None
        self._token = self.read_token()

    def read_token(self):
        token = self._config[self.name]["TOKEN"]
        
        try:
            token_expiration = datetime.strptime(self._config[self.name]["TOKEN_EXPIRATION"], "%Y-%m-%d %H:%M:%S")
        except:
            return self.get_access_token(update=True)
        
        if token is None:
            return self.get_access_token(update=True)
        
        elif (token_expiration-datetime.now()).total_seconds() < 0:
            return self.get_access_token(update=True)
    
        return token
    
    def get_app(self):
        return self._app
    
    def get_secret(self):
        return self._secret
    
    def get_access_token(self, update=False):
        endpoint = "oauth2/tokenP"
    
        data = {
            'grant_type': 'client_credentials',
            'appkey': self.get_app(),
            'appsecret': self.get_secret(),
        }
        
        try:
            res = self.post(endpoint=endpoint, data=data)
            TOKEN_EXPIRATION = res.json()["access_token_token_expired"]
            
            token = f"{res.json()['token_type']} {res.json()['access_token']}"
            if update:
                self._config[self.name]["TOKEN"] = token
                self._config[self.name]["TOKEN_EXPIRATION"] = TOKEN_EXPIRATION
                
                with open(self._config_path, "r+") as f:
                    config = yaml.safe_load(f)
                    config[self.name]["TOKEN"] = token
                    config[self.name]["TOKEN_EXPIRATION"] = TOKEN_EXPIRATION
                    
                with open(self._config_path, "w+") as f:
                    yaml.dump(config, f)
                
                print(f"{datetime.now()} - {self.name} - Access Token Updated")
                print(f"\tResponse URL: {res.url}")
                
                
            return token # 접근 토큰을 반환
        except requests.RequestException as e:
            return {'error': str(e)}
        
    def get_approval(self):
        endpoint = "oauth2/Approval"
        data = {
            'grant_type': 'client_credentials',
            'appkey': self.get_app(),
            'secretkey': self.get_secret(),
        }
        res = self.post(endpoint, data=data)

        approval_key = res.json()["approval_key"]
        print(f"{datetime.now()} - {self.name} - Approval Response Status: {res.status_code}")
        print(f"\tResponse URL: {res.url}")

        return approval_key
    

    def check_holiday(self, base_date: str):
        res = self.get("uapi/domestic-stock/v1/quotations/chk-holiday", 
                 headers={
                     "Authorization": self._token,
                     "appkey": self._app,
                     "appsecret": self._secret,
                     "tr_id": "CTCA0903R",
                     "custtype": "P"},
                 params={
                     "BASS_DT": base_date,
                     "CTX_AREA_NK": "",
                     "CTX_AREA_FK": ""
                 })
        
        return res.json()["output"]

    def get_candle(self, code, cont=False, base_time=None):
        endpoint = "uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        
        headers = {
            "authorization": self.read_token(),
            "appkey": self.get_app(),
            "appsecret": self.get_secret(),
            "tr_id" :  "FHKST03010200",
            "tr_cont": "" if not cont else "N",
            "custtype": "P"
        }
        
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_HOUR_1": base_time if base_time is not None else datetime.now().strftime("%HH%MM%SS") if datetime.now().time() < time(15, 30, 0) else "153000",
            "FID_PW_DATA_INCU_YN": "Y"
        }

        try:
            res = self.get(endpoint=endpoint, params=params, headers=headers)
            return res

        except Exception as e:
            return {"error": str(e)}
        
    def day_candle(self, code, FID_INPUT_DATE_2=datetime.now()):
        FID_INPUT_DATE_1=(FID_INPUT_DATE_2 - timedelta(days=100))
        
        endpoint = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        
        headers = {
            "authorization": self.read_token(),
            "appkey": self.get_app(),
            "appsecret": self.get_secret(),
            "tr_id" :  "FHKST03010100",
            "custtype": "P"
        }
    
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": FID_INPUT_DATE_1.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": FID_INPUT_DATE_2.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0"
        }

        try:
            res = self.get(endpoint=endpoint, params=params, headers=headers)
            return res

        except Exception as e:
            return {"error": str(e)}