# antenna/utils/web.py
# 提供三項網路工具：
#   1. Email        — SMTP 寄信包裝（with-context 管理連線生命週期）
#   2. get_local_ip — 取本機區網 IP（用於組實驗資料夾名稱 / 寄信標題）
#   3. connect_network_drive — 用 Windows `net use` 掛載 NAS 磁碟機
from loguru import logger
from typing import Union, Sequence, Self

#* Email
from smtplib import SMTP
from email.mime.text import MIMEText
class Email(SMTP):
    """SMTP 寄信包裝，繼承自 smtplib.SMTP。

    設計原則
    --------
    * 以 **with-context** 方式使用（``with Email(...) as e:``），確保
      不論是否發生例外，``__exit__`` 都會呼叫 ``self.quit()`` 關閉連線。
    * 一個 ``Email`` 實例對應一次 SMTP session，建構時就完成登入，
      避免多次驗證的延遲與帳號鎖定風險。

    前置條件
    --------
    * 寄件帳號使用 Google **應用程式密碼**（App Password），而非一般
      Gmail 密碼。若 Google 帳號開啟兩步驟驗證，必須在
      https://myaccount.google.com/apppasswords 產生 16 碼密碼。
    * 使用 gmail.com SMTP (port 587 + STARTTLS)；若主機防火牆封鎖
      對外 TCP 587，連線將逾時（Training Server 需確認防火牆規則）。
    * ``cc``/``bcc`` 預設值為 ``[]``（可變物件），呼叫端不應直接對
      這兩個預設 list 做 in-place 修改，否則下次呼叫會帶入殘留資料。
    """

    def __init__(
        self,
        to:Union[str, list[str]],
        cc:list = [], bcc:list = [],
        from_addr_pwd:tuple = ("ailab@ee.ccu.edu.tw", "bung ovhd rrcu nayg")
    ) -> None:
        """
        Args:
            to (str, Sequence[str]): Target Address
            cc: 副本收件人 email
            bcc: 密件副本收件人 email

        Example:
            ```
            with Email("weiwen@alum.ccu.edu.tw") as email:

                msg = email.getText(
                    'AILAB Antenna Notice',
                    "This is a test email sent from Python."
                )
                status = email.sendMessage(msg.as_string())

                if status == {}:
                    print("Email sent successfully!")
                else:
                    print('Email send failed!')
            ```

        Reference
        ---------
        https://steam.oxxostudio.tw/category/python/example/gmail.html
        """
        super().__init__("smtp.gmail.com", 587)  #! port 587 需防火牆允許對外連線
        self.starttls()                           #? STARTTLS 將明文連線升級為加密通道，login 前必須先執行
        self.login(from_addr_pwd[0], from_addr_pwd[1])  #! 密碼為 Google App Password，非帳號原始密碼

        self.to_list = to if isinstance(to, list) else [to]
        self.cc_list = cc if isinstance(cc, list) else [cc]
        self.bcc_list = bcc if isinstance(bcc, list) else [bcc]

        self.all_recipients = self.to_list + self.cc_list + self.bcc_list  #* sendmail 的實際收件人清單（含 bcc）
        self.from_addr = from_addr_pwd[0]

    def getText(self, subject:str = 'AILAB Antenna Notice', message:str = "" , from_name:str = "AILAB Antenna Team"):
        """組裝 MIME 純文字訊息物件。

        注意
        ----
        * ``MIMEText`` 預設編碼為 ``us-ascii``；若 ``message`` 含中文，
          SMTP 傳輸時可能出現亂碼。如需中文內文，可改為
          ``MIMEText(message, 'plain', 'utf-8')``。
        * ``bcc`` 收件人**不**寫入 header（符合 BCC 慣例），
          但已列入 ``self.all_recipients`` 讓 ``sendmail`` 實際投遞。
        * 呼叫後會將結果存入 ``self.msg_str``，供後續 ``sendMessage``
          無參數呼叫時使用。
        """
        msg = MIMEText(message)

        msg['Subject'] = subject
        msg['From'] = from_name or str(self.from_addr)
        msg['To'] = ", ".join(self.to_list)
        msg['Cc'] = ", ".join(self.cc_list)

        self.msg_str = msg.as_string()
        return msg

    def sendMessage(self, message:str = None):
        """透過已開啟的 SMTP session 傳送郵件。

        參數
        ----
        message : str, optional
            若省略，使用 ``getText`` 所快取的 ``self.msg_str``。
            若直接傳入 raw message 字串，則以此為準。

        回傳
        ----
        dict
            ``sendmail`` 回傳值：成功時為 ``{}``，失敗的收件人
            會以 ``{addr: (code, msg)}`` 形式記錄在字典中。
        """
        assert self.all_recipients, "Please select sender."
        return self.sendmail(self.from_addr, self.all_recipients, message or self.msg_str)

    def __enter__(self) -> Self:
        # 支援 with-context；直接回傳自身，不重複初始化
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb) -> None:
        # 不論是否發生例外，都必須關閉 SMTP 連線以釋放資源
        self.quit()



from socket import socket, AF_INET, SOCK_DGRAM
def get_local_ip():
    """取得本機在區域網路的 IP 位址。

    實作原理
    --------
    開啟一個 UDP socket 並「連線」到 Google DNS (8.8.8.8:80)；
    UDP 不會真的發送封包，但 OS 會根據路由表決定應使用哪張網卡，
    ``getsockname()`` 即可取得該網卡的 IP。
    此方法不依賴 DNS 解析，也不受 ``127.0.0.1`` 的回環介面干擾。

    使用場景
    --------
    * 訓練腳本以此 IP 組合實驗資料夾名稱，使多台機器同時訓練時
      結果目錄不互相覆蓋。
    * 寄信通知的標題也可帶入 IP，方便辨識是哪台機器完成訓練。

    陷阱
    ----
    * 若主機完全無網路（例如測試環境），``connect`` 可能拋出例外，
      此時回傳 ``"127.0.0.1"`` 並以 loguru error 記錄原因。
    * 多網卡主機（如同時插有實驗室有線網路與 VPN）會依預設路由
      決定回傳哪張網卡的 IP，結果可能不一致。

    Returns
    -------
    str
        本機 IP 字串，失敗時為 ``"127.0.0.1"``。
    """
    s = socket(AF_INET, SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Google DNS — 僅用於觸發路由決策，不實際傳送封包
        ip = s.getsockname()[0]
    except Exception as e:
        ip = "127.0.0.1"            #! 無網路時的 fallback，訓練資料夾名稱可能因此重複
        logger.error(e)
    finally:
        s.close()
    return ip


import subprocess
from os.path import exists
def connect_network_drive(drive_letter, network_path, user="", password="", *, del_old = False, verbose:bool = False):
    """
    Checks if a network drive is connected and attempts to connect it if not.
    This version includes optional user and password authentication.

    Args:
        drive_letter (str): The drive letter to connect, e.g., "T:".
        network_path (str): The UNC path of the network share, e.g., r"\\140.123.106.219\temp".
        user (str): The username for authentication. Defaults to an empty string.
        password (str): The password for authentication. Defaults to an empty string.

    Returns:
        bool: True if the connection is successful or already exists, False otherwise.

    實作說明
    --------
    此函式透過 Windows 內建指令 ``net use`` 掛載 SMB/CIFS 網路磁碟。

    前置條件
    --------
    * 僅適用於 Windows 環境（依賴 ``net use`` 指令）。
    * 若 NAS 需要帳密，必須傳入 ``user`` 與 ``password``；
      ``net use`` 的參數順序為 ``<drive> <path> <password> /user:<user>``，
      注意密碼在 user 前面（``command_args`` 的組裝已按此順序）。
    * ``/persistent:yes`` 使磁碟掛載在使用者登入後自動重新連線；
      若不需要持久化，可移除此旗標。

    陷阱與注意事項
    --------------
    * ``del_old=True``：先執行 ``net use <drive> /delete`` 強制清除舊連線。
      若磁碟原本不存在，``CalledProcessError`` 會被靜默忽略（``pass``）。
    * ``exists(drive_letter)``：``subprocess.run`` 拋出例外後，額外以
      ``os.path.exists`` 確認磁碟是否其實已可存取（例如已被其他程序掛載）；
      若可存取則視為成功，避免重複掛載造成錯誤。
    * ``shell=True`` 搭配 ``check=True``：在 Windows 上 ``net use`` 需要
      透過 shell 執行才能正確解析路徑；但 ``shell=True`` 有注入風險，
      呼叫端應確保 ``drive_letter``、``network_path``、``user``、``password``
      來自可信來源（config 檔或環境變數），不可接受外部使用者輸入。
    * 若 NAS IP 不通（防火牆、NAS 關機），指令會 timeout，訓練啟動將卡住。
      建議在 CI / 自動化腳本中加上 timeout 保護。
    """

    if del_old:
        try:
            subprocess.run(
                ['net', 'use', drive_letter, '/delete'], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            pass  #? 磁碟不存在時 /delete 會失敗，直接忽略即可

    # Build the net use command.
    command_args = ['net', 'use', drive_letter, network_path, '/persistent:yes']
    if user and password:
        command_args.extend([password, '/user:' + user])  #! net use 參數順序：密碼在 /user 前

    # Attempt to connect the network drive.
    try:
        if verbose:
            logger.info(f"Attempting to connect to `{drive_letter}` ...")
        subprocess.run(command_args, check=True, shell=True, capture_output=True, text=True)  #* shell=True 讓 Windows 正確解析 UNC 路徑
        logger.success(f"Network drive `{drive_letter}` successfully connected.")
        return True

    except subprocess.CalledProcessError as e:
        if exists(drive_letter): # Check if the drive is already connected.
            if verbose:
                logger.info(f"Network drive `{drive_letter}` is already connected. Skipping connection.")
            return True
        else:
            logger.warning(f"Connection failed: {e.stderr}")
        return False
