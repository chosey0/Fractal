from PyQt5.QtCore import QThread, pyqtSignal
import asyncio
import websockets
from datetime import datetime, timezone
import json

class WebSocketWorker(QThread):
    message_received = pyqtSignal(list)

    def __init__(self, name, approval):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.name = name
        
        self.config_message_queue = asyncio.Queue()
        self._approval = approval
        
        if self.name == "real":
            self.url = "ws://ops.koreainvestment.com:21000"
        elif self.name == "simulation":
            self.url = "ws://ops.koreainvestment.com:31000"
        
    async def receive_and_process(self):
        print(f'{datetime.now()} - {self.name} - ws_agent.py/receive_and_process - Websocket Connected')
        print(f"\tSession Host: {self.session.host}")
        print(f"\tSession Port: {self.session.port}")
        print(f"\tAgent Websocket URL: {self.url}")

        while True:
            try:
                if not self.config_message_queue.empty():
                    sendmsg = await self.config_message_queue.get()
                    await self.subscribe_func(sendmsg)
                    
                data = await self.session.recv()
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
                        await self.session.pong(data)
                    # else:
                    #     await response_handler(jsonObject)
                
            except websockets.ConnectionClosed as e:
                print(f"[Error][{datetime.now()}] - Connection closed: {e}")
                break
            
            except Exception as e:
                print(f"[Error][{datetime.now()}] - {e}")
                continue
            
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - [Agent] ws_agent.py/receive_and_process 종료')
    
    async def emit_wrapper(self, data):
        self.message_received.emit(data)
    
    async def connect(self):
        async with websockets.connect(self.url, ping_interval=30) as session:
            # await self.subscribe_func(session)
            self.session = session
            await self.receive_and_process()
    
        
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
    async def subscribe_func(self, senddata):
        # for senddata in self.sendlist:
        await self.session.send(senddata)
        await asyncio.sleep(0.5)
        print(f"Input Command is :{senddata}")


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