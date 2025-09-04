腳本 (Script)
==========

正常情況下是不能直接執行的，所以可以用以下方式解決。

1. 在Python檔的開頭加入下列程式碼
    ```python
    from sys import path
    from os.path import dirname, join
    path.append(join(dirname(__file__),'..'))
    ```
2. 在終端機 (根目錄) 執行以下指令
    ```bash
    python -m script.<腳本名>

    # Example
    python -m script.kill
    ```