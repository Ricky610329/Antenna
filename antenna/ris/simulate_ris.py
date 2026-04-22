"""RIS（Reconfigurable Intelligent Surface）模擬介面。

以解析式陣列因子 (Array Factor, AF) 計算 RIS 的遠場響應，
不依賴 HFSS COM，屬於純 numpy/torch 實作。
"""

import numpy as np
import torch
from numpy import cos, deg2rad, pi, sin

from antenna.utils import *

# --- RIS 幾何與入射波預設參數（28 GHz 毫米波場景） ---
# ? 若需調整不同頻段 / 反射場景，可於實例化時覆寫 __init__ 參數
_DEFAULT_FREQ_HZ = 28e9  # 工作頻率：28 GHz
_DEFAULT_FEED_DIST = 500e-3  # 饋源到 RIS 中心距離：500 mm
_DEFAULT_INC_TH_DEG = -40.0  # 入射波 theta（度）
_DEFAULT_INC_PH_DEG = 90.0  # 入射波 phi（度）
_SPEED_OF_LIGHT = 3e8  # 光速 (m/s)


class RISSimulator:
    """以陣列因子公式計算 RIS 遠場響應的可微模擬器。"""

    def __init__(
        self,
        element_num: int,
        freq_hz: float = _DEFAULT_FREQ_HZ,
        feed_distance_m: float = _DEFAULT_FEED_DIST,
        inc_theta_deg: float = _DEFAULT_INC_TH_DEG,
        inc_phi_deg: float = _DEFAULT_INC_PH_DEG,
    ):
        """
        :param element_num: 單邊元素數（RIS 總元素數 = element_num ** 2）
        :param freq_hz: 工作頻率（Hz）
        :param feed_distance_m: 饋源到 RIS 中心距離（m）
        :param inc_theta_deg: 入射波 theta 角（度）
        :param inc_phi_deg: 入射波 phi 角（度）
        """
        self.element_num = element_num
        self.freq_hz = freq_hz
        self.feed_distance_m = feed_distance_m
        self.inc_theta_deg = inc_theta_deg
        self.inc_phi_deg = inc_phi_deg
        self.pre_calAF = self._calAF()

    def __str__(self):
        return f"RISSimulator(element_num={self.element_num})"

    def _calAF(self):
        """預先計算與相位控制無關的 Array Factor 分量（只與幾何 / 入射波相關）。"""
        f = self.freq_hz
        R = self.feed_distance_m
        de = 0.5 * _SPEED_OF_LIGHT / f  # 元素間距 = 半波長
        k = 2 * pi / (_SPEED_OF_LIGHT / f)  # 波數
        element_num = self.element_num

        M = np.arange(1, element_num + 1)
        N = np.arange(1, element_num + 1)

        # 遠場觀測方向的取樣網格（theta: -90~90 度、phi: 0~360 度）
        theDeg = np.arange(-90, 90.1, 0.5)
        phiDeg = np.arange(0, 361, 2)

        theta = deg2rad(theDeg)
        phi = deg2rad(phiDeg)

        THETA, PHI = np.meshgrid(theta, phi)
        THETA = np.round(THETA, decimals=4)
        PHI = np.round(PHI, decimals=4)
        u = np.round(sin(THETA) * cos(PHI), decimals=4)
        v = np.round(sin(THETA) * sin(PHI), decimals=4)

        inc_th_rad = deg2rad(self.inc_theta_deg)
        inc_ph_rad = deg2rad(self.inc_phi_deg)

        # RIS 元素位於 x/y 平面上的中心化座標
        low_bound = -(element_num / 2 - 0.5)
        high_bound = low_bound + element_num
        x, y = np.mgrid[low_bound:high_bound, low_bound:high_bound]

        # 饋源位置（球座標 -> 直角座標）
        feed_x = R * sin(inc_th_rad) * cos(inc_ph_rad)
        feed_y = R * sin(inc_th_rad) * sin(inc_ph_rad)
        feed_z = R * cos(inc_th_rad)

        det_x = feed_x - x * de
        det_y = feed_y - y * de
        det_z = feed_z

        Ri = np.sqrt(det_x**2 + det_y**2 + det_z**2)
        incPD = k * Ri  # 入射相位差

        # 以 RIS 中心為原點的元素索引
        mm = M - (element_num + 1) / 2
        nn2 = N - (element_num + 1) / 2
        m, n = np.meshgrid(mm, nn2)
        m = m.reshape(1, 1, -1)
        n = n.reshape(1, 1, -1)

        incphase = incPD.transpose(1, 0).reshape(1, 1, -1)

        # 原本以 list comprehension 走訪扁平後的 ndarray，等同於增加一個 size=1 的軸，直接 reshape 即可
        u3 = u.reshape(*u.shape, 1)
        v3 = v.reshape(*v.shape, 1)

        ui = np.tile(u3, element_num**2) * m
        vi = np.tile(v3, element_num**2) * n
        ## element_num^2 為 RIS 總元素數（例如 10x10 -> 100）

        sptfun = (ui + vi).reshape(u.shape[0], u.shape[1], -1)

        incphase = incphase.astype(np.float32)
        incphase_c = incphase + 0.0j

        pre_calAF = torch.tensor(
            np.exp(1j * (-incphase_c)) * np.exp(1j * k * de * sptfun),
            dtype=torch.complex64,  # 視需求可改為 complex128
            device=config.device,
        )

        return pre_calAF

    def __call__(self, pattern: Tensor):
        """輸入 RIS 相位 pattern（0~1，代表 0~pi 相位），輸出遠場 dB 響應。"""
        MPD = pattern * torch.pi  # 保留梯度，不做 detach

        MPD = MPD.reshape(self.element_num, self.element_num)

        refphase = MPD.t().reshape(1, 1, -1)  # 轉置後展平
        refphase_c = refphase + 0.0j

        af = self.pre_calAF.to(refphase_c.device) * torch.exp(1j * refphase_c)
        AF = torch.abs(torch.sum(af, dim=2))  # shape: (1, 361)

        mag = torch.max(AF, dim=1, keepdim=True).values  # shape: (1, 1)

        # 避免 log(0)
        AF = torch.clamp(AF, min=1e-8)
        mag = torch.clamp(mag, min=1e-8)
        dB_AF = 20 * torch.log10(AF / mag)

        return {"response": dB_AF[0]}
