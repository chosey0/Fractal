import requests as req
import yaml


def order_exec():
    base = ""
    url = "/websocket/stock"
    with open("config.yaml", encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    headers = {
        "token": config["TOKEN"],
        "tr_type": "3"
    }
    body = {
        "tr_cd": "SC1",
        "tr_key": ""
    }