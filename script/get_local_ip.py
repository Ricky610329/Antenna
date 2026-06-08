"""
get_local_ip.py — 取得本機對外網卡的 IPv4 位址。

原理：
    利用 UDP socket 「假連線」到外部公共位址 (Google DNS 8.8.8.8:80)。
    UDP 連線不發送任何實際封包，但作業系統會選擇對應外部路由的本機網卡，
    getsockname() 即可讀出該網卡被分配的 IPv4 位址。

回傳格式：
    字串，例如 "192.168.1.42"；若發生例外（無網路、socket 錯誤）
    則回傳例外訊息字串。

專案用途：
    天線反向設計閉迴路系統以本機區網 IP 末段作為實驗識別碼，
    用於區分多台機器產生的資料夾名稱與 log 標籤，
    例如末段 "42" → 實驗目錄 run_42_xxx。

環境陷阱：
    - 多網卡（有線 + 無線 + VPN）：OS 路由表決定「對外」介面，
      回傳的 IP 可能不是你預期的那張網卡。
    - 完全無網路時：connect() 可能丟出 OSError，
      此函式會以 str(e) 回傳錯誤文字而非 IP，
      呼叫端若直接取末段數字將得到非預期結果，需自行驗證格式。
    - 127.0.0.1 / loopback：正常情況下不會回傳 loopback，
      但某些極度受限的容器環境例外。
"""

from socket import (
    socket,
    AF_INET,
    SOCK_DGRAM
)
def getLocalIP():
    """
    取得本機對外路由所使用之網卡的 IPv4 位址。

    方法：以 UDP（SOCK_DGRAM）對 8.8.8.8:80 執行不送封包的假連線，
    讓 OS 選定對外網卡後，透過 getsockname() 讀取該介面的 IP。

    Returns
    -------
    str
        成功時為形如 "x.x.x.x" 的 IPv4 位址字串；
        失敗時為例外的字串表示（非 IP 格式）。
    """
    try:
        # 創建一個 socket 連接到一個公共的 DNS 服務器
        s = socket(AF_INET, SOCK_DGRAM)  ###* AF_INET=IPv4, SOCK_DGRAM=UDP（不會真的送封包）
        s.connect(("8.8.8.8", 80))       #? 假連線：OS 依路由表選定對外網卡，不發送任何資料
        local_ip = s.getsockname()[0]    #* getsockname() 回傳 (ip, port)，取 [0] 得 IP 字串
        s.close()                        #! 務必關閉 socket，避免描述符洩漏
        return local_ip
    except Exception as e:
        return str(e)                    #! 無網路或 socket 失敗時回傳例外訊息，呼叫端須自行檢查格式

if __name__ == "__main__":
    print(f"Local IP Address: {getLocalIP()}")  # 直接執行時印出本機 IP，供手動確認
