import requests as req
import yaml
from datetime import datetime, timedelta

def get_access_token(name, update=False):
    with open("env.yaml", encoding='UTF-8') as f:
        config = yaml.safe_load(f)

    EBEST_APP = config[name]["APP_KEY"]
    EBEST_SECRET = config[name]["SECRET_KEY"]
    
    url = "https://openapi.ls-sec.co.kr:8080/oauth2/token"
    headers = {
        'content-type': 'application/x-www-form-urlencoded'
        }
    param = {
                "appkey": EBEST_APP,
                "appsecretkey": EBEST_SECRET,
                "grant_type": "client_credentials",
                "scope": "oob"
            }
    try:
        res = req.post(url, headers=headers, data=param)
        TOKEN_EXPIRATION = datetime.strftime(datetime.now() + timedelta(seconds=int(res.json()["expires_in"])), "%Y-%m-%d %H:%M:%S")
        
        token = f"{res.json()['token_type']} {res.json()['access_token']}"
        if update:
            config[name]["TOKEN"] = token
            config[name]["TOKEN_EXPIRATION"] = TOKEN_EXPIRATION
            
            with open("env.yaml", "r+") as f:
                config = yaml.safe_load(f)
                config[name]["TOKEN"] = token
                config[name]["TOKEN_EXPIRATION"] = TOKEN_EXPIRATION
                
            with open("env.yaml", "w+") as f:
                yaml.dump(config, f)
            
            print(f"{datetime.now()} - {name} - Access Token Updated")
            print(f"\tResponse URL: {res.url}")
            
        return token # 접근 토큰을 반환
    except req.RequestException as e:
        return {'error': str(e)}

def read_token(name):
    with open("env.yaml", encoding='UTF-8') as f:
        config = yaml.safe_load(f)
        try:
            token = config[name]["TOKEN"]
            token_expiration = config[name]["TOKEN_EXPIRATION"]
        except:
            return get_access_token(name, update=True)
        del config
    
    try:
        token_expiration = datetime.strptime(token_expiration, "%Y-%m-%d %H:%M:%S")
    except:
        return get_access_token(name, update=True)
    
    if token is None:
        return get_access_token(name, update=True)
    
    elif (token_expiration-datetime.now()).total_seconds() < 0:
        return get_access_token(name, update=True)

    return token