
from socket import (
    socket,
    AF_INET,
    SOCK_DGRAM
)
def getLocalIP():
    try:
        # 創建一個 socket 連接到一個公共的 DNS 服務器
        s = socket(AF_INET, SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    print(f"Local IP Address: {getLocalIP()}")