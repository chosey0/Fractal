from PyQt5.QtCore import QThread, pyqtSignal
import asyncio
import websockets
from datetime import datetime, timezone
import json
import ccxt.pro as ccxtpro
import ccxt
import pandas as pd
import numpy as np
import toolz.itertoolz as tz

class WebSocketWorker(QThread):
    message_received = pyqtSignal(list)

    def __init__(self, name, approval):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.name = name
        self.setObjectName(name)
        self.config_message_queue = asyncio.Queue()
        self._approval = approval
        
        if self.name == "real":
            self.url = "ws://ops.koreainvestment.com:21000"
        elif self.name == "simulation":
            self.url = "ws://ops.koreainvestment.com:31000"
        
    async def receive_and_process(self, session):
        print(f'{datetime.now()} - {self.name} - ws_agent.py/receive_and_process - Websocket Connected')
        print(f"\tSession Host: {session.host}")
        print(f"\tSession Port: {session.port}")
        print(f"\tAgent Websocket URL: {self.url}")

        while True:
            try:
                if not self.config_message_queue.empty():
                    sendmsg = await self.config_message_queue.get()
                    await self.subscribe_func(session, sendmsg)

                data = await session.recv()
                recv_time = datetime.now(tz=timezone.utc).timestamp()
                
                if data[0] == '0':
                    # data[15:21]
                    await self.emit_wrapper([recv_time, data])
                    
                elif data[0] == '1':
                    continue
                else:
                    jsonObject = json.loads(data)
                    trid = jsonObject["header"]["tr_id"]
                    
                    if trid == "PINGPONG":
                        print(f"[PINGPONG RECV][{datetime.now()}] - [{data}]")
                        await session.pong(data)
                    else:
                        await response_handler(jsonObject)
                
            except websockets.ConnectionClosed as e:
                print(f"[Error][{datetime.now()}] - Connection closed: {e}")
                break
                # continue
            
            except Exception as e:
                print(f"[Error][{datetime.now()}] - {e}")
                continue
            
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - [Agent] ws_agent.py/receive_and_process 종료')
    
    async def emit_wrapper(self, data):
        self.recv_datastr(data, self.message_received.emit)
    
    async def connect(self):
        async with websockets.connect(self.url, ping_interval=30) as session:
            # await self.subscribe_func(session)
            self.session = session
            await self.receive_and_process(session)
        
    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.connect())

    # 웹소켓 데이터 구독 등록/해제 요청 String 생성 함수
    def configure_transaction(self, code):
        TRANSACTION_DETAIL = {
                "1": ("H0STASP0", "1"), "2": ("H0STASP0", "2"),
                "3": ("H0STCNT0", "1"), "4": ("H0STCNT0", "2"),
                "5": ("H0STCNI0", "1"), "6": ("H0STCNI0", "2"),
                "7": ("H0STCNI9", "1"), "8": ("H0STCNI9", "2"),
            }
        
        cmd = "3" #input(f"{code} - TRID 선택 : ") # KIS Github 150 line 참고
        
        if cmd == "0":
            print(f"[KIS][SETTING][{datetime.now()}]: - Connection Terminated\n")
        
        if cmd in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            tr_id, tr_type = TRANSACTION_DETAIL.get(cmd, ("", "")) # 사용자 입력에 따라 TRID와 TR Type을 맵핑해서 반환
            message = '{"header":{"approval_key": "%s","custtype":"P","tr_type":"%s","content-type":"utf-8"},"body":{"input":{"tr_id":"%s","tr_key":"%s"}}}'%(self._approval, tr_type, tr_id, code)
            disconnect_message = '{"header":{"approval_key": "%s","custtype":"P","tr_type":"%s","content-type":"utf-8"},"body":{"input":{"tr_id":"%s","tr_key":"%s"}}}'%(self._approval, "2", tr_id, code)
            
            return tr_id, message, disconnect_message

        else:
            print("> Wrong Input: ", cmd)
            retry = input("Do you want to enter again? (y/n): ")
            if retry.lower() == "y":
                self.configure_transaction(code)
            
            
# 구독 등록/해제 요청 함수
    async def subscribe_func(self, session, senddata):
        # for senddata in self.sendlist:
        await session.send(senddata)
        await asyncio.sleep(0.5)
        print(f"Input Command is :{senddata}")
        
    def recv_datastr(self, recv_data, callback):
        recv_time, datastr = recv_data
        recvstr = datastr.split('|')
        
        # 수신 데이터의 수량(나노 초 수준의 고빈도 데이터의 경우 여러건이 수신될 수 있음)
        n_item = int(recvstr[2])
        
        # 수신 데이터 전문
        data = recvstr[-1].split("^")
        
        if n_item == 1:
            callback([recv_time, data[0], data[1], data[2]])
        else:
            temp = list(tz.partition(len(data)//n_item, data))
            chunks = [[(recv_time + idx*1e-3)]+list(chunk) for idx, chunk in enumerate(temp)]
            [callback([chunk[0], chunk[1], chunk[2], chunk[3]]) for chunk in chunks]

# 웹소켓 응답 처리 함수        
async def response_handler(jsonObject):
    rt_cd = jsonObject["body"].get("rt_cd", "")
    
    if rt_cd == '1':
        if jsonObject["body"].get("msg1") != 'ALREADY IN SUBSCRIBE':
            print(f"[ERROR][Runner][{datetime.now()}]: [{jsonObject['header']['tr_key']}][{rt_cd}]\n"
                    f"[MSG] {jsonObject['body']['msg1']}")
            
    elif rt_cd == '0':
        if "tr_key" in jsonObject["header"] and "body" in jsonObject:
            print(f"[RESPONSE][{jsonObject['header']['tr_key']}][{datetime.now()}] - MSG: {jsonObject['body']['msg1']}")
            
    else:
        print(f"[WARNING][{datetime.now()}] - Unexpected response format {jsonObject}")



class UpbitWorker(WebSocketWorker):
    def __init__(self):
        super().__init__(None, None)
        self.exchange = ccxtpro.upbit()
    
    def configure_transaction(self, symbol):
        return None, symbol+"/KRW", None
    
    async def connect(self):
        await self.receive_and_process()
        
    async def receive_and_process(self):
        while True:
            trade_data = await self.exchange.watch_trades(symbol="ZRO/KRW")
            
            code = trade_data["info"]["code"]
            price = trade_data["info"]["trade_price"]
            date = trade_data["info"]["trade_date"].replace("-", "")
            time = trade_data["info"]["trade_time"].replace(":", "")
            
            recv_time = datetime.now(tz=timezone.utc).timestamp()

            await self.emit_wrapper([recv_time, code, time, price])
            
    def prev_data(self, symbol="BTC/KRW", timeframe='1d'):
        exchange = ccxt.upbit(config={
                'apiKey': "RkxRNxRhefbbmhOi5ND2SPFdwBCqxMWpi8r8Ht5U",
                'secret': "jdx40SDcOQPvY8JPVvbK8JLRbxOCKyrNjuHwsbZ6",
                'enableRateLimit': True
            }
        )
        ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe)

        df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        pd_ts = pd.to_datetime(df['Time'], utc=True, unit='ms')     # unix timestamp to pandas Timeestamp
        pd_ts = pd_ts.dt.tz_convert("Asia/Seoul")                       # convert timezone
        pd_ts = pd_ts.dt.tz_localize(None)
        df.set_index(pd_ts, inplace=True)
        df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index(inplace=True)
        df["ma20"] = df["Close"].rolling(window=20).mean()
        df["ma120"] = df["Close"].rolling(window=120).mean()

        df.set_index(pd.DatetimeIndex(df["Time"]).as_unit("ms").asi8, inplace=True)
    
        df = df[["Time", "Open", "High", "Low", "Close", "ma20", "ma120"]].dropna()
        
        df["low_point"] = [np.nan] * len(df)
        df["high_point"] = [np.nan] * len(df)
        df["low_prob"] = [np.nan] * len(df)
        df["high_prob"] = [np.nan] * len(df)
        df["none_prob"] = [np.nan] * len(df)
        
        return df.to_dict(orient="index")