"""
antenna — 微帶貼片天線「反向設計」核心套件。

層級：
    pattern.py / response.py   資料抽象 (圖樣 / 響應)
    training.py                config 驅動的訓練核心 (run_training)
    zoo.py                     模型動物園 (config 用名字選 GEN/SM)
    models/                    模型 (外殼 / 生成器 / 代理模型)
    optim/                     優化 (Ranger / ACP 排程器)
    losses.py                  可製造性損失 + R_feed 指標
    monitor.py                 TensorBoard 監控
    patch/                     HFSS 模擬器 (COM)
    utils/                     工具底座 (Record / size_converter 等核心工具)
    legacy/                    隔離的舊資料層 (Data / DataManager；非核心)

Example::

    from antenna import AntennaPattern, AntennaResponse, TargetResponse, get_result_path
    from antenna.utils import config
    spec = TargetResponse(labels=('S11', 'Gain'), x='n257')
    AntennaResponse.use(spec)
"""
from antenna.response import AntennaResponse, MultiResponses, TargetResponse
from antenna.pattern import AntennaPattern
from antenna.utils.run_setup import get_result_path
