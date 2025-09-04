from antenna.utils import *
import numpy as np
from numpy import ndarray, sin, cos, pi, array, deg2rad, rad2deg
import torch 
if np.__version__ >= '1.25.0':
    from numpy import round as np_round
else:
    from numpy import round_ as np_round # type: ignore

class RISSimulator:
    def __init__(self, element_num):
        """
        :param element_num: element_num = config.element_num
        """
        self.element_num = element_num
        self.pre_calAF = self._calAF()

    def __str__(self):
        return f"RISSimulator(element_num={self.element_num})"
    def _calAF(self):
        """Calculate Array Factor"""
        c = 3e8  # 光速
        f = 28e9 #頻率
        lamda = c / f
        R = 500e-3
        de = 0.5 * c / f 
        k = 2 * pi / (c / f) #角波束
        element_num = self.element_num

        M = np.arange(1,element_num+1)
        N = np.arange(1,element_num+1)

        theDeg = [i for i in np.arange(-90,90.1,0.5)]
        phiDeg = [i for i in range(0,361,2)]

        incPH_deg , refPH_deg = 90 , 0
        incTH_deg , refTH_deg = -40 , 20 #先不要改

        theta = deg2rad(theDeg)
        phi   = deg2rad(phiDeg)

        [THETA, PHI]=np.meshgrid(theta,phi) # 使用 PyTorch meshgrid
        THETA = np_round(THETA , decimals = 4)
        PHI = np_round(PHI , decimals = 4)
        u = np_round(((sin(THETA)) * (cos(PHI))) , decimals = 4)
        v = np_round(((sin(THETA)) * (sin(PHI))) , decimals = 4)

        incTH_rad, incPH_rad = deg2rad([incTH_deg, incPH_deg])

        low_bound_X = -(element_num / 2 - 0.5)
        high_bound_X = low_bound_X + element_num

        low_bound_Y = -(element_num / 2 - 0.5)
        high_bound_Y = low_bound_Y + element_num
        x, y = np.mgrid[low_bound_Y:high_bound_Y, low_bound_X:high_bound_X]

        feed_x = R * sin(incTH_rad) * cos(incPH_rad)
        feed_y = R * sin(incTH_rad) * sin(incPH_rad)
        feed_z = R * cos(incTH_rad)

        det_x = feed_x - x * de
        det_y = feed_y - y * de
        det_z = feed_z

        Ri = np.sqrt(det_x ** 2 + det_y ** 2 + det_z ** 2)
        incPD = k * Ri

        mm = M - (element_num+1)/2
        nn2 = N - (element_num+1)/2
        [m,n]=np.meshgrid(mm,nn2)

        m = np.reshape(m,(1,1,-1))
        n = np.reshape(n,(1,1,-1))

        incphase = np.transpose(incPD, axes=(1, 0)).reshape(1, 1, -1)

        c_temp_1 = np.array([e.tolist() for e in u.flatten()]).reshape(u.shape[0],u.shape[1],-1)
        c_temp_2 = np.array([e.tolist() for e in v.flatten()]).reshape(v.shape[0],v.shape[1],-1)

        ui = (np.tile(c_temp_1,(element_num**2))) * m
        vi = (np.tile(c_temp_2,(element_num**2))) * n
        ## element_num^2是指面鏡大小 ， 若面鏡為 10*10， element_num則是100

        sptl=ui+vi
        sptfun = np.reshape(sptl, (u.shape[0], u.shape[1], -1))

        incphase = incphase.astype(np.float32)
        incphase_c = incphase + 0.0j

        pre_calAF = torch.tensor(
            np.exp(1j * (-incphase_c)) * np.exp(1j * k * de * sptfun),
            dtype=torch.complex64,  # or complex128, 看你需要的精度
            device=config.device
        )

        return pre_calAF

    def __call__(self, pattern:Tensor):
        MPD = pattern * torch.pi  # no detach
        
        MPD = MPD.reshape(config.element_num, config.element_num)
        
        refphase = MPD.t().reshape(1, 1, -1)  # transpose and reshape
        refphase_c = refphase + 0.0j  # complex

        af = self.pre_calAF.to(refphase_c.device) * torch.exp(1j * refphase_c)  # still complex
        AF = torch.abs(torch.sum(af, dim=2))  # shape: (1, 361)
        
        mag = torch.max(AF, dim=1, keepdim=True).values  # shape: (1, 1)

        AF = torch.clamp(AF, min=1e-8)
        mag = torch.clamp(mag, min=1e-8) # avoid log(0)
        dB_AF = 20 * torch.log10(AF / mag)

        return {'response': dB_AF[0]}