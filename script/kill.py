from argparse import  ArgumentParser
from  psutil import (
    process_iter,
    NoSuchProcess, 
    AccessDenied, 
    ZombieProcess
)
from loguru import logger

def kill(process_name='ansysedt.exe'):
    for proc in process_iter(['pid', 'name']):
        if process_name.lower() in proc.info['name'].lower():
            try:
                proc.kill()  # 結束進程
                logger.warning(f"Process {process_name} terminated.")
            except (NoSuchProcess, AccessDenied, ZombieProcess):
                pass


if __name__ == "__main__":
    parser = ArgumentParser(
        description = "用來結束HFSS的工作階段"
    )

    parser.add_argument(
        "--name",
        type=str,
        default = r"ansysedt.exe",
        help = "This is process name."    
    )

    args =parser.parse_args()
    kill(args.name)