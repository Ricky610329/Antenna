from loguru import logger
from typing import Union, Sequence, Self

#* Email
from smtplib import SMTP
from email.mime.text import MIMEText
class Email(SMTP):
   
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
        super().__init__("smtp.gmail.com", 587)
        self.starttls()
        self.login(from_addr_pwd[0], from_addr_pwd[1])
        
        self.to_list = to if isinstance(to, list) else [to]
        self.cc_list = cc if isinstance(cc, list) else [cc]
        self.bcc_list = bcc if isinstance(bcc, list) else [bcc]

        self.all_recipients = self.to_list + self.cc_list + self.bcc_list 
        self.from_addr = from_addr_pwd[0]
    
    def getText(self, subject:str = 'AILAB Antenna Notice', message:str = "" , from_name:str = "AILAB Antenna Team"):
        msg = MIMEText(message)
        
        msg['Subject'] = subject
        msg['From'] = from_name or str(self.from_addr)
        msg['To'] = ", ".join(self.to_list)
        msg['Cc'] = ", ".join(self.cc_list)

        self.msg_str = msg.as_string()
        return msg
                        
    def sendMessage(self, message:str = None):
        assert self.all_recipients, "Please select sender."
        return self.sendmail(self.from_addr, self.all_recipients, message or self.msg_str)

    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb) -> None:
        self.quit()



from socket import socket, AF_INET, SOCK_DGRAM
def get_local_ip():
    s = socket(AF_INET, SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Google DNS
        ip = s.getsockname()[0]
    except Exception as e:
        ip = "127.0.0.1"
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
    """
    
    if del_old:
        try:
            subprocess.run(
                ['net', 'use', drive_letter, '/delete'], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            pass

    # Build the net use command.
    command_args = ['net', 'use', drive_letter, network_path, '/persistent:yes']
    if user and password:
        command_args.extend([password, '/user:' + user])

    # Attempt to connect the network drive.
    try:
        if verbose:
            logger.info(f"Attempting to connect to `{drive_letter}` ...")
        subprocess.run(command_args, check=True, shell=True, capture_output=True, text=True)
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