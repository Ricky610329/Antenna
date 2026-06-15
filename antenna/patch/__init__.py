"""
antenna/patch — 微帶貼片天線的 HFSS 模擬器套件 (SIM 角色)。

純模擬器出口：SinglePortSimulator / DualPortSimulator / com_error。
(響應損失函數已歸位到 antenna/losses.py。)
"""
from .patch_simulator import com_error
from .patch_simulator.dual_port import DualPortSimulator      # 雙埠 HFSS 模擬器
from .patch_simulator.single_port import SinglePortSimulator  # 單埠 HFSS 模擬器
from .patch_simulator.single_port_rad import SinglePortRadSimulator  # 單埠 + 方向圖萃取 (不侵入擴充)
