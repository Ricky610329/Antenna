"""日月光「歷來實驗總整理」第五版：各代方法的數學式，以 matplotlib mathtext 渲染成 PNG。

用法（repo 根目錄）：
    python script/figs/report_ase_math.py [--only g1_chain,g4_wm]
輸出：docs/進度報告/10月_歷來實驗總整理/figs/math/<代號>.png，白底、dpi 300。

全份字級一致：每個式子都以 FS pt、Computer Modern 字型渲染，投影片產生器 make_pptx_v5.py 依 dpi 換回原尺寸貼上，
所以投影片上的式子都是同一個字級，不縮放。一個式子可以有一到兩行，各行分開渲染再靠左疊起來，行距固定。
寬度上限 MAX_W 吋，對應數學式頁中欄寬；超過就中止，要改成兩行。

式子與符號定義抄自同夾 notes/gen12_縮寫與數學式.md、gen3_縮寫與數學式.md、gen4_縮寫與數學式.md，
每式都回碩論式號、演算法行號或程式核對過；哪些是簡化表示、省略了什麼，寫在投影片備忘稿。零 NAS。
"""
import argparse
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "進度報告" / "10月_歷來實驗總整理" / "figs" / "math"
FS = 18                 # pt，全份一致
DPI = 300
INK = "#152C3D"
LINE_GAP = 0.08         # 兩行之間的空白，吋
MAX_W = 6.05            # 數學式頁中欄可用寬，吋

# 代號：(一行或兩行 mathtext)
FORMULAS = {
    # ---------------------------------------------------------------- 第一代，陳廷茂碩論
    "g1_chain": [r"$\dfrac{\partial\mathcal{L}}{\partial\theta}=\dfrac{\partial\mathcal{L}}{\partial\hat{y}}\,"
                 r"\dfrac{\partial\hat{y}}{\partial x}\,\dfrac{\partial x}{\partial\theta}$",
                 r"$\hat{y}=\mathrm{GEN}_{\phi}(x),\quad x=\mathrm{MPGN}_{\theta}(s^{*})$"],
    "g1_loss": [r"$\mathcal{L}=\dfrac{1}{N}\sum_{n=1}^{N}\max(R_n-C_R,\,0)^2"
                r"+\dfrac{1}{N}\sum_{n=1}^{N}\min(G_n-C_G,\,0)^2$"],
    "g1_ste": [r"$q=\mathrm{Sign}(z),\quad b=(1+q)/2,\quad "
               r"\partial\mathcal{L}/\partial z=\partial\mathcal{L}/\partial q$"],
    "g1_online": [r"$\mathcal{D}\leftarrow\mathcal{D}\cup\{(x_t,\,y_t)\}$",
                  r"$\mathrm{MSE}_{\mathcal{D}}(\phi)=\mathrm{mean}\,(\mathrm{GEN}_{\phi}(x)-y)^2<10^{-4}$"],
    # ---------------------------------------------------------------- 第二代，錢鵬予碩論
    "g2_bsn": [r"$\hat{x}=\dfrac{x}{M}\ \ (x>0),\qquad \hat{x}=\dfrac{x}{|m|}\ \ (x<0)$",
               r"$M\leftarrow\alpha M+(1-\alpha)\max(x),\ \ m\leftarrow\alpha m+(1-\alpha)\min(x)$"],
    "g2_loss": [r"$\mathcal{L}=\sum_{j=1}^{4}\mathbf{1}[\hat{S}_j\notin\Omega_j]\cdot\mathrm{SmoothL1}(\hat{S}_j-S_j)$"],
    "g2_flip": [r"$\hat{y}_i\leftarrow 1-\hat{y}_i,\quad i\in I_k,\quad |I_k|=k=\mu\cdot\mathrm{len}(\hat{y})$"],
    "g2_rollback": [r"$y^{*}\leftarrow\hat{y}\ \ \mathrm{if}\ \ \mathcal{L}(\hat{y})<\mathcal{L}_{\min}$",
                    r"$\hat{y}\leftarrow\mathrm{mutate}(y^{*},\mu)\ \ \mathrm{when}\ \ t\ \mathrm{mod}\ c=0,"
                    r"\quad \mu\leftarrow\lambda\mu$"],
    # ---------------------------------------------------------------- 第三代，吳維文碩論
    "g3_acp": [r"$\eta=\eta_{\mathrm{min}}+\Delta\eta\cdot\Phi(T_{\mathrm{cur}}),\quad "
               r"\tau=\tau_{\mathrm{min}}+\Delta\tau\cdot\Phi(T_{\mathrm{cur}})$",
               r"$c_p\geq P\ \Rightarrow\ T_i\leftarrow\max(\gamma_{\mathrm{decay}}\,T_i,\ T_0/2)$"],
    "g3_tau": [r"$P_{\mathrm{soft}}=\sigma\left(\dfrac{\tilde{P}-k}{\tau}\right)$"],
    "g3_loss": [r"$\mathcal{L}_{\mathrm{EM}}=\ell\,(\hat{S}_{11}[\,\hat{S}_{11}>-10\,],\,{-10})"
                r"+\ell\,(\hat{G}[\,\hat{G}<4\,],\,4)$",
                r"$\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{EM}}+w_{\mathrm{SC}}\cdot\mathcal{L}_{\mathrm{SC}}$"],
    "g3_sc": [r"$\mathcal{L}_{\mathrm{SC}}=1/\lambda_2(L),\quad L=D-A$",
              r"$A_{ij}=p_i\,p_j+\varepsilon\ \ \mathrm{if}\ \ i\sim j,\qquad A_{ij}=0\ \ \mathrm{otherwise}$"],
    "g3_dlf": [r"$B_{\mathrm{train}}=\{\,(P_i,\,y_i,\,\mathcal{L}_i)\in B\ :\ \mathcal{L}_i\leq\bar{\mathcal{L}}_t\,\}$",
               r"$\bar{\mathcal{L}}_t=\mathrm{mean}(\mathcal{H})$"],
    # ---------------------------------------------------------------- 第四代，程式核對
    "g4_wm": [r"$W=\min_{b}\ \min_{f\in F_b}\ s_b\,[\,y_b(f)-t_b\,]$"],
    "g4_rad": [r"$R=\min_{\phi}\ \left[\ \min_{|\theta|\leq 45\degree}\,"
               r"g_\phi(\theta)-g_\phi(0\degree)+3\ \right]$"],
    "g4_select": [r"$\mathcal{S}_t=\mathrm{top}_K\,\{\,\mu_t(x)-\kappa\,\sigma_t(x)\ :\ x\in\mathcal{C}_t\,\}$",
                  r"$\mathcal{D}_{t+1}=\mathcal{D}_t\cup\{\,(x,\ y_{\mathrm{HFSS}}(x))\ :\ x\in\mathcal{S}_t\,\}$"],
    "g4_delta": [r"$\hat{y}(x)=y_{\mathrm{HFSS}}(x_c)+\Delta_\psi\left(x_c,\ x\oplus x_c,\ y_{\mathrm{HFSS}}(x_c)\right)$"],
    "g4_rank": [r"$\rho=\mathrm{corr}\left(\mathrm{rank}(\hat{W}),\ \mathrm{rank}(W)\right)$"],
}


def _render_line(tex):
    fig = plt.figure(figsize=(12, 2), dpi=DPI)
    fig.text(0.01, 0.5, tex, fontsize=FS, color=INK, va="center", ha="left")
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    bio.seek(0)
    return Image.open(bio).convert("RGB")


def render(key, lines):
    ims = [_render_line(t) for t in lines]
    gap = round(LINE_GAP * DPI)
    w = max(im.width for im in ims)
    h = sum(im.height for im in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for im in ims:
        out.paste(im, (0, y))
        y += im.height + gap
    OUT.mkdir(parents=True, exist_ok=True)
    out.save(OUT / f"{key}.png", dpi=(DPI, DPI))
    return w / DPI, h / DPI


def main():
    plt.rcParams["mathtext.fontset"] = "cm"
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(FORMULAS))
    args = ap.parse_args()
    too_wide = []
    for key in args.only.split(","):
        w, h = render(key, FORMULAS[key])
        print(f"{key:12s} {w:5.2f} × {h:4.2f} 吋")
        if w > MAX_W:
            too_wide.append(key)
    if too_wide:
        raise SystemExit(f"超過 {MAX_W} 吋，要改成兩行：{too_wide}")


if __name__ == "__main__":
    main()
