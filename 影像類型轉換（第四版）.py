"""影像轉換系統（第四版）
第三版：灰階、純黑白、調色盤、X 光、中值濾波、DCT、拉普拉斯銳利化。
第四版新增：
  1. 三種筆觸：畫筆、橡皮擦、麥克筆，可以用滑鼠直接在「轉換結果」上作畫
  2. GIF 動畫：一次選擇多張圖（不限張數），依檔名順序做成 GIF 動畫
     （講義 Ch02：GIF 為 256 色調色盤影像、以 LZW 壓縮，動畫是把多張圖疊成一個檔）
轉換、作畫或製作 GIF 後可按「儲存成新檔」存檔。

需要套件：Pillow、NumPy  (pip install pillow numpy)
"""

from abc import ABC, abstractmethod
import math
import os
import re
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk


# ---------- 檔案讀寫 ----------

class ImageFileManager:
    def load(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def save(self, img: Image.Image, path: str) -> None:
        img.save(path)


# ---------- 影像轉換 ----------

class ImageConverter(ABC):
    @abstractmethod
    def convert(self, img: Image.Image) -> Image.Image:
        pass


class GrayscaleConverter(ImageConverter):
    def convert(self, img: Image.Image) -> Image.Image:
        return img.convert("L")


class BinaryConverter(ImageConverter):
    def __init__(self, threshold: int = 128):
        self.__threshold = threshold

    def convert(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        # 亮度 >= 門檻值變白，其餘變黑
        return gray.point(lambda p: 255 if p >= self.__threshold else 0).convert("1")


class PaletteConverter(ImageConverter):
    def __init__(self, colorCount: int = 256):
        self.__colorCount = colorCount

    def convert(self, img: Image.Image) -> Image.Image:
        return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=self.__colorCount)


class XrayConverter(ImageConverter):
    def convert(self, img: Image.Image) -> Image.Image:
        # X 光效果：先轉灰階，再把明暗反轉（負片）
        return ImageOps.invert(img.convert("L"))


# ---------- HW3 新增：中值濾波 ----------

class MedianFilterConverter(ImageConverter):
    """中位數濾波器：G(x, y) = Median{ I(x, y) | (x, y) ∈ M }

    依照講義 p.40 的三種情況處理：
      (a) 角落像素：只排序自己和鄰近 3 個像素（共 4 個）
      (b) 邊緣像素：只排序自己和鄰近 5 個像素（共 6 個）
      (c) 內部像素：正常排序遮罩內 9 個像素
    影像外的位置不處理（不補值）。
    """

    def __init__(self, size: int = 3):
        if size % 2 == 0:
            raise ValueError("遮罩大小必須是奇數")
        self.__size = size

    def convert(self, img: Image.Image) -> Image.Image:
        # 彩色影像：R、G、B 三個通道分開各自做中值濾波，再合併回彩色
        rgb = np.array(img.convert("RGB"), dtype=np.float64)
        channels = [self.__filter(rgb[:, :, c]) for c in range(3)]
        return Image.fromarray(np.stack(channels, axis=2), mode="RGB")

    def __filter(self, channel: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        r = self.__size // 2
        # 用 NaN 補在影像外圍，代表「影像外的不處理」
        padded = np.pad(channel, r, mode="constant", constant_values=np.nan)

        # 把遮罩內每個位置的值疊成 (size*size, h, w)，一次算出所有像素的中位數
        windows = np.stack([padded[dy:dy + h, dx:dx + w]
                            for dy in range(self.__size)
                            for dx in range(self.__size)])
        # nanmedian 會忽略 NaN：角落只剩 4 個值、邊緣剩 6 個、內部 9 個
        result = np.nanmedian(windows, axis=0)
        return np.clip(np.round(result), 0, 255).astype(np.uint8)


# ---------- HW3 新增：DCT 轉換 ----------

class DCTProcessor:
    """以 8×8 block 為單位做離散餘弦轉換（講義公式 (1)(2)）。

    D(i,j) = 1/√(2N) · C(i)C(j) · ΣΣ f(x,y) cos[(2x+1)iπ/2N] cos[(2y+1)jπ/2N]
    f(x,y) = 1/√(2N) · ΣΣ C(i)C(j) D(i,j) cos[(2x+1)iπ/2N] cos[(2y+1)jπ/2N]
    其中 C(0) = 1/√2，其餘 C = 1；f(x,y) 是像素值減 128 後的結果。
    """

    N = 8

    def __init__(self):
        n = self.N
        # 先把 cos 項做成 N×N 的表：T[i][x] = C(i) · cos[(2x+1)iπ / 2N]
        self.__table = np.zeros((n, n))
        for i in range(n):
            c = 1 / math.sqrt(2) if i == 0 else 1.0
            for x in range(n):
                self.__table[i, x] = c * math.cos((2 * x + 1) * i * math.pi / (2 * n))
        self.__scale = 1 / math.sqrt(2 * n)

    def forward(self, block: np.ndarray) -> np.ndarray:
        """FDCT：空間域 block（已減 128）→ 頻率域係數"""
        # 雙重 Σ 拆成矩陣乘法：D = scale · T · f · Tᵀ
        return self.__scale * self.__table @ block @ self.__table.T

    def inverse(self, coeff: np.ndarray) -> np.ndarray:
        """IDCT：頻率域係數 → 空間域 block（之後要再加 128）"""
        return self.__scale * self.__table.T @ coeff @ self.__table

    def process(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """整張灰階影像切成不重疊的 8×8 block，
        每塊做 FDCT → 把 mask 為 True 的係數歸 0 → IDCT。"""
        n = self.N
        h, w = gray.shape
        # 寬高不是 8 的倍數時，先用邊緣像素補齊，最後再裁掉
        ph, pw = (-h) % n, (-w) % n
        img = np.pad(gray.astype(np.float64), ((0, ph), (0, pw)), mode="edge")
        out = np.empty_like(img)

        for by in range(0, img.shape[0], n):
            for bx in range(0, img.shape[1], n):
                block = img[by:by + n, bx:bx + n] - 128       # 每個像素先減 128
                coeff = self.forward(block)
                coeff[mask] = 0                                # 指定頻段的係數歸 0
                out[by:by + n, bx:bx + n] = self.inverse(coeff) + 128   # 還原後加回 128

        out = out[:h, :w]
        return np.clip(np.round(out), 0, 255).astype(np.uint8)


class DCTConverter(ImageConverter):
    """DCT 轉換後把某一個頻段的係數歸 0，再反轉換回影像。

    以係數位置 (i, j) 的 i + j 判斷頻率高低（8×8 block 中 i + j 介於 0~14）：
      DC  ：i + j = 0          （只有左上角 1 個係數，代表整塊的平均亮度）
      低頻：1 ≤ i + j ≤ 4      （14 個係數，不歸 0）
      中頻：5 ≤ i + j ≤ 8      （28 個係數）
      高頻：9 ≤ i + j ≤ 14     （21 個係數，代表細節與邊緣）
    """

    BANDS = {
        "dc":   (0, 0),
        "mid":  (5, 8),
        "high": (9, 14),
    }

    def __init__(self, band: str):
        if band not in self.BANDS:
            raise ValueError(f"未知的頻段：{band}")
        low, high = self.BANDS[band]
        i, j = np.indices((DCTProcessor.N, DCTProcessor.N))
        self.__mask = (i + j >= low) & (i + j <= high)
        self.__processor = DCTProcessor()

    def convert(self, img: Image.Image) -> Image.Image:
        gray = np.array(img.convert("L"))
        return Image.fromarray(self.__processor.process(gray, self.__mask), mode="L")


# ---------- 新增：拉普拉斯濾波器（銳利濾波） ----------

class LaplacianConverter(ImageConverter):
    """Laplacian 銳利濾波（講義 p.46–52）。

    ∇²f(x,y) = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) − 4f(x,y)   公式 (4)

    兩種遮罩（p.50）：
      圖 C 垂直水平方向：[[0,1,0],[1,-4,1],[0,1,0]]
      圖 D 對角方向    ：[[1,0,1],[0,-4,0],[1,0,1]]

    兩種輸出（p.52 圖 E）：
      edge   ：(b) 經過 Laplacian 遮罩處理後的結果（取絕對值顯示邊緣）
      sharpen：(c) 把 (b) 跟原圖結合 → g = f − ∇²f
               （遮罩中心為負，所以用「減」才會加強邊緣）
    """

    KERNELS = {
        "hv":   np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float64),
        "diag": np.array([[1, 0, 1],
                          [0, -4, 0],
                          [1, 0, 1]], dtype=np.float64),
    }

    def __init__(self, direction: str = "hv", output: str = "sharpen"):
        if direction not in self.KERNELS:
            raise ValueError(f"未知的遮罩方向：{direction}")
        if output not in ("edge", "sharpen"):
            raise ValueError(f"未知的輸出類型：{output}")
        self.__kernel = self.KERNELS[direction]
        self.__output = output

    def convert(self, img: Image.Image) -> Image.Image:
        # 講義用灰階影像示範，所以先轉灰階；用 float 計算才能保留負值
        gray = np.array(img.convert("L"), dtype=np.float64)
        lap = self.__convolve(gray)

        if self.__output == "edge":
            result = np.abs(lap)                 # (b) 邊緣：正負變化都顯示成亮線
        else:
            result = gray - lap                  # (c) 銳利化：原圖 − Laplacian

        return Image.fromarray(np.clip(np.round(result), 0, 255).astype(np.uint8), mode="L")

    def __convolve(self, gray: np.ndarray) -> np.ndarray:
        """3×3 遮罩在每個像素上做「對應位置相乘再加總」。
        影像外圍用邊緣像素補一圈，讓邊界像素也能套用遮罩。"""
        h, w = gray.shape
        k = self.__kernel.shape[0]
        padded = np.pad(gray, k // 2, mode="edge")
        out = np.zeros_like(gray)
        # 遮罩的每一格 (dy, dx)：把整張影像對應的位移版本乘上權重後加總
        for dy in range(k):
            for dx in range(k):
                weight = self.__kernel[dy, dx]
                if weight != 0:
                    out += weight * padded[dy:dy + h, dx:dx + w]
        return out


# ---------- 第四版新增：三種筆觸 ----------

class DrawingCanvas:
    """作畫用的畫布：底圖 + 一層透明的「筆跡圖層」。

    筆跡都畫在圖層上，不會改到底圖，所以橡皮擦只要把圖層擦成透明，
    就會露出底下原本的圖。最後把圖層疊回底圖就是作畫結果。
    """

    def __init__(self, base: Image.Image):
        self.base = base.convert("RGB")
        self.layer = Image.new("RGBA", self.base.size, (0, 0, 0, 0))  # 筆跡圖層（全透明）
        self.buffer = None        # 麥克筆畫到一半時的暫存圖層
        self.bufferOpacity = 1.0  # 暫存圖層疊上去時的透明度

    def clear(self) -> None:
        """清除全部筆跡"""
        self.layer = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        self.buffer = None

    def composite(self) -> Image.Image:
        """底圖 + 筆跡圖層（+ 正在畫的麥克筆）疊成一張 RGB 影像"""
        out = self.base.convert("RGBA")
        out.alpha_composite(self.layer)
        if self.buffer is not None:
            out.alpha_composite(applyOpacity(self.buffer, self.bufferOpacity))
        return out.convert("RGB")


def applyOpacity(img: Image.Image, opacity: float) -> Image.Image:
    """把 RGBA 影像的透明度（alpha）整體乘上 opacity"""
    result = img.copy()
    alpha = result.getchannel("A").point(lambda a: int(a * opacity))
    result.putalpha(alpha)
    return result


class Brush(ABC):
    """筆觸的共同介面：按下滑鼠 begin → 拖曳時一段一段 drawSegment → 放開 end"""

    def begin(self, canvas: DrawingCanvas) -> None:
        pass

    @abstractmethod
    def drawSegment(self, canvas: DrawingCanvas, p0, p1, color, width: int) -> None:
        pass

    def end(self, canvas: DrawingCanvas) -> None:
        pass

    @staticmethod
    def _roundLine(img: Image.Image, p0, p1, fill, width: int) -> None:
        """畫一段線，兩端補圓點，讓連續的線段接起來是平滑的"""
        draw = ImageDraw.Draw(img)
        draw.line([p0, p1], fill=fill, width=width)
        r = width / 2
        for x, y in (p0, p1):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


class PenBrush(Brush):
    """畫筆：不透明、細的實心線條"""

    def drawSegment(self, canvas, p0, p1, color, width):
        self._roundLine(canvas.layer, p0, p1, tuple(color) + (255,), width)


class EraserBrush(Brush):
    """橡皮擦：把筆跡圖層擦成透明，露出底下原本的圖（不會擦到底圖）"""

    SIZE_RATE = 2  # 橡皮擦比畫筆粗一點比較好擦

    def drawSegment(self, canvas, p0, p1, color, width):
        # ImageDraw 在 RGBA 影像上是「直接寫入」不是混色，所以填 (0,0,0,0) 就是擦成透明
        self._roundLine(canvas.layer, p0, p1, (0, 0, 0, 0), width * self.SIZE_RATE)


class MarkerBrush(Brush):
    """麥克筆：粗、半透明的線條，疊上去後底圖還看得到。

    同一筆畫來回重疊的地方不會越塗越深：
    先把這一筆完整畫在暫存圖層，放開滑鼠時才用半透明疊到筆跡圖層。
    """

    OPACITY = 0.4
    SIZE_RATE = 3  # 麥克筆筆頭比畫筆粗

    def begin(self, canvas):
        canvas.buffer = Image.new("RGBA", canvas.base.size, (0, 0, 0, 0))
        canvas.bufferOpacity = self.OPACITY

    def drawSegment(self, canvas, p0, p1, color, width):
        if canvas.buffer is None:
            self.begin(canvas)
        self._roundLine(canvas.buffer, p0, p1, tuple(color) + (255,), width * self.SIZE_RATE)

    def end(self, canvas):
        if canvas.buffer is not None:
            canvas.layer.alpha_composite(applyOpacity(canvas.buffer, self.OPACITY))
            canvas.buffer = None


# ---------- 第四版新增：GIF 動畫 ----------

class GifMaker:
    """把多張影像做成 GIF 動畫（講義 Ch02 p.52–57）。

    - 每一格都轉成 256 色的調色盤影像（GIF 最多只能 256 色）
    - 存檔時 Pillow 會用 LZW 壓縮每一格
    - loop=0 代表無限循環播放
    """

    BACKGROUND = (255, 255, 255)

    @staticmethod
    def sortByFileName(paths):
        """依檔名排序；檔名裡的數字照數值比，所以 2.png 會排在 10.png 前面"""
        def key(path):
            name = os.path.basename(path).lower()
            return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]
        return sorted(paths, key=key)

    def makeFrames(self, images):
        """所有圖統一成第一張圖的大小（等比例縮放，不夠的地方補白邊），再轉成 256 色"""
        if not images:
            raise ValueError("沒有影像可以製作 GIF")
        size = images[0].size
        frames = []
        for img in images:
            fitted = ImageOps.pad(img.convert("RGB"), size, color=self.BACKGROUND)
            frames.append(fitted.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
        return frames

    def save(self, frames, path: str, durationMs: int) -> None:
        frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:],
                       duration=durationMs, loop=0)


# ---------- 第四版新增：自訂游標 ----------

class BrushCursor:
    """自訂滑鼠游標：畫出畫筆／橡皮擦／麥克筆的小圖示。

    macOS 的 Tk 不能載入自訂游標檔，所以改成把圖示直接畫在畫布的滑鼠位置上，
    再把系統箭頭隱藏起來，看起來就像游標變成了筆。
    圖示斜 45 度朝右上，筆尖（熱點）在左下角，筆尖就是實際下筆的位置。
    """

    SIZE = 34
    TIP = (2, 31)   # 筆尖在圖示中的位置
    OUTLINE = (40, 40, 40, 255)

    def __init__(self):
        self.__cache = {}

    def icon(self, name: str, color) -> Image.Image:
        key = (name, tuple(color))
        if key not in self.__cache:
            img = Image.new("RGBA", (self.SIZE, self.SIZE), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            getattr(self, f"_draw_{name}")(draw, tuple(color) + (255,))
            self.__cache[key] = img
        return self.__cache[key]

    def drawOn(self, img: Image.Image, name: str, color, pos) -> None:
        """把圖示貼到 img 上，讓筆尖剛好對準滑鼠位置 pos"""
        icon = self.icon(name, color)
        x, y = int(pos[0]) - self.TIP[0], int(pos[1]) - self.TIP[1]
        img.paste(icon, (x, y), icon)

    # 沿著筆身方向的座標：t = 離筆尖多遠，o = 往筆身兩側偏多少
    def _pt(self, t, o):
        c = math.sqrt(0.5)
        return (self.TIP[0] + t * c + o * c, self.TIP[1] - t * c + o * c)

    def _quad(self, draw, t0, t1, w0, w1, fill):
        draw.polygon([self._pt(t0, -w0), self._pt(t1, -w1),
                      self._pt(t1, w1), self._pt(t0, w0)], fill=fill, outline=self.OUTLINE)

    def _draw_pen(self, draw, color):
        self._quad(draw, 9, 33, 3.5, 3.5, (250, 200, 40, 255))      # 黃色筆身
        self._quad(draw, 33, 38, 3.5, 3.5, (240, 150, 160, 255))    # 尾端小橡皮擦
        self._quad(draw, 3, 9, 1.2, 3.5, (235, 200, 150, 255))      # 削過的木頭
        draw.polygon([self._pt(0, 0), self._pt(3.5, -1.3), self._pt(3.5, 1.3)],
                     fill=color, outline=self.OUTLINE)               # 筆芯＝目前顏色

    def _draw_eraser(self, draw, color):
        self._quad(draw, 0, 11, 6, 6, (245, 160, 175, 255))         # 粉紅色擦頭
        self._quad(draw, 11, 30, 6, 6, (70, 120, 200, 255))         # 藍色紙套
        self._quad(draw, 17, 23, 6, 6, (255, 255, 255, 255))        # 紙套上的白條

    def _draw_marker(self, draw, color):
        draw.polygon([self._pt(0, -1.5), self._pt(5, -3), self._pt(5, 3), self._pt(1, 2.5)],
                     fill=color, outline=self.OUTLINE)               # 斜切筆頭＝目前顏色
        self._quad(draw, 5, 9, 3.5, 3.5, (150, 150, 150, 255))      # 灰色筆頸
        self._quad(draw, 9, 36, 5.5, 5.5, (60, 60, 60, 255))        # 深色筆身
        self._quad(draw, 27, 31, 5.5, 5.5, color)                    # 筆身上的顏色環


# ---------- 使用者介面 ----------

class MainWindow:
    PREVIEW_SIZE = (380, 380)

    def __init__(self, root: tk.Tk):
        self.__root = root
        self.__originalImage = None
        self.__convertedImage = None
        self.__fileManager = ImageFileManager()
        self.__converter = None
        self.__originalPath = ""
        self.__suffix = ""
        self.__photos = {}  # 保留參考，避免預覽圖被回收

        # 第四版：作畫用的狀態
        self.__fullCanvas = None      # 原尺寸畫布（存檔用）
        self.__previewCanvas = None   # 預覽尺寸畫布（畫面顯示用，畫起來比較順）
        self.__previewScale = 1.0     # 原圖寬 ÷ 預覽圖寬，用來把滑鼠座標換算回原圖
        self.__lastPoint = None
        self.__color = (255, 0, 0)
        self.__brushes = {"pen": PenBrush(), "eraser": EraserBrush(), "marker": MarkerBrush()}
        self.__brushCursor = BrushCursor()
        self.__mousePos = None        # 滑鼠在畫布上的位置（畫游標圖示用）
        self.__hasStrokes = False     # 左邊有沒有畫過東西

        # 第四版：GIF 動畫的狀態
        self.__gifMaker = GifMaker()
        self.__gifFrames = None
        self.__gifThumbs = []
        self.__gifPaths = []
        self.__gifIndex = 0       # 下一格要顯示第幾張
        self.__gifShown = 0       # 目前畫面上是第幾張（停止時就是停住的那一格）
        self.__animJob = None

        root.title("影像轉換系統（第四版）")
        root.resizable(False, False)

        # 上方按鈕列（第一列：HW2 原本的功能）
        bar = tk.Frame(root, padx=10, pady=10)
        bar.pack(fill="x")
        tk.Button(bar, text="選擇彩色影像", width=14,
                  command=self.onSelectImage).pack(side="left", padx=4)

        self.__convertButtons = []
        for text, mode in (("轉換成灰階", "grayscale"),
                           ("轉換成純黑白", "binary"),
                           ("轉換成調色盤影像", "palette"),
                           ("轉換成 X 光影像", "xray")):
            btn = tk.Button(bar, text=text, state="disabled",
                            command=lambda m=mode: self.onConvert(m))
            btn.pack(side="left", padx=4)
            self.__convertButtons.append(btn)

        # 最右邊：儲存按鈕，轉換完成後才能按
        self.__saveButton = tk.Button(bar, text="儲存成新檔", width=12,
                                      state="disabled", command=self.onSave)
        self.__saveButton.pack(side="right", padx=4)

        # 第二列：HW3 新增的功能
        bar2 = tk.Frame(root, padx=10)
        bar2.pack(fill="x", pady=(0, 10))

        tk.Label(bar2, text="濾波：").pack(side="left")
        btn = tk.Button(bar2, text="中值濾波", state="disabled",
                        command=lambda: self.onConvert("median"))
        btn.pack(side="left", padx=4)
        self.__convertButtons.append(btn)

        tk.Label(bar2, text="遮罩大小").pack(side="left", padx=(6, 0))
        self.__medianSize = tk.Spinbox(bar2, values=(3, 5, 7, 9), width=3,
                                       state="readonly")
        self.__medianSize.pack(side="left", padx=(2, 16))

        tk.Label(bar2, text="DCT：").pack(side="left")
        for text, mode in (("DC 係數歸 0", "dct_dc"),
                           ("中頻係數歸 0", "dct_mid"),
                           ("高頻係數歸 0", "dct_high")):
            btn = tk.Button(bar2, text=text, state="disabled",
                            command=lambda m=mode: self.onConvert(m))
            btn.pack(side="left", padx=4)
            self.__convertButtons.append(btn)

        # 第三列：拉普拉斯濾波器
        bar3 = tk.Frame(root, padx=10)
        bar3.pack(fill="x", pady=(0, 10))

        tk.Label(bar3, text="拉普拉斯：").pack(side="left")
        self.__lapDirection = tk.StringVar(value="hv")
        tk.Radiobutton(bar3, text="垂直水平遮罩", value="hv",
                       variable=self.__lapDirection).pack(side="left")
        tk.Radiobutton(bar3, text="對角遮罩", value="diag",
                       variable=self.__lapDirection).pack(side="left", padx=(0, 10))
        for text, mode in (("Laplacian 邊緣", "lap_edge"),
                           ("Laplacian 銳利化", "lap_sharpen")):
            btn = tk.Button(bar3, text=text, state="disabled",
                            command=lambda m=mode: self.onConvert(m))
            btn.pack(side="left", padx=4)
            self.__convertButtons.append(btn)

        # 第四列（第四版新增）：GIF 動畫
        bar4 = tk.Frame(root, padx=10)
        bar4.pack(fill="x", pady=(0, 10))

        tk.Label(bar4, text="GIF：").pack(side="left")
        tk.Button(bar4, text="選擇多張圖製作 GIF",
                  command=self.onMakeGif).pack(side="left", padx=4)
        tk.Label(bar4, text="每張停留").pack(side="left", padx=(12, 0))
        self.__gifDuration = tk.StringVar(value="500")
        tk.Spinbox(bar4, from_=50, to=5000, increment=50, width=5,
                   textvariable=self.__gifDuration).pack(side="left", padx=2)
        self.__gifDuration.set("500")   # Spinbox 建立時可能會把值重設成 from_，這裡再設回預設值
        tk.Label(bar4, text="ms").pack(side="left")
        self.__playButton = tk.Button(bar4, text="⏸ 停止", width=8, state="disabled",
                                      command=self.onTogglePlay)
        self.__playButton.pack(side="left", padx=(12, 0))
        # 「以結果作為原圖」：GIF 停住的那一格、或轉換／作畫的結果，都可以拿來當原圖繼續處理
        self.__chainButton = tk.Button(bar4, text="以結果作為原圖", state="disabled",
                                       command=self.onUseResult)
        self.__chainButton.pack(side="left", padx=4)
        self.__gifCount = tk.Label(bar4, text="已選 0 張")
        self.__gifCount.pack(side="left", padx=12)

        # 第五列（第四版新增）：三種筆觸
        bar5 = tk.Frame(root, padx=10)
        bar5.pack(fill="x", pady=(0, 10))

        tk.Label(bar5, text="筆觸：").pack(side="left")
        # 預設不選任何筆觸（用 "none"；如果用空字串，Tk 會把三個選項都顯示成「部分選取」的一槓）
        self.__brushName = tk.StringVar(value="none")
        for text, name in (("畫筆", "pen"), ("橡皮擦", "eraser"), ("麥克筆", "marker")):
            tk.Radiobutton(bar5, text=text, value=name, variable=self.__brushName,
                           command=self.onSelectBrush).pack(side="left")

        tk.Button(bar5, text="選顏色", command=self.onPickColor).pack(side="left", padx=(12, 2))
        # 用 Label 當色塊（macOS 的按鈕不會顯示背景色）
        self.__colorSwatch = tk.Label(bar5, width=2, bg=self.__toHex(self.__color),
                                      relief="solid", bd=1)
        self.__colorSwatch.pack(side="left", padx=(0, 12))

        tk.Label(bar5, text="粗細").pack(side="left")
        self.__brushSize = tk.Scale(bar5, from_=1, to=30, orient="horizontal",
                                    length=120, showvalue=True)
        self.__brushSize.set(4)
        self.__brushSize.pack(side="left", padx=(2, 12))

        tk.Button(bar5, text="清除筆跡", command=self.onClearDrawing).pack(side="left", padx=4)

        # 預覽區：左邊原始影像（也是作畫的畫布）、右邊轉換結果／GIF 動畫
        preview = tk.Frame(root, padx=10)
        preview.pack()
        self.__panels = {}
        for col, (key, title) in enumerate((("original", "原始影像"),
                                            ("converted", "轉換結果"))):
            tk.Label(preview, text=title).grid(row=0, column=col, pady=(0, 4))
            box = tk.Frame(preview, width=self.PREVIEW_SIZE[0],
                           height=self.PREVIEW_SIZE[1], bg="#dddddd")
            box.grid(row=1, column=col, padx=6)
            box.pack_propagate(False)
            # 邊框、內距都設 0，滑鼠座標才會剛好等於預覽圖上的像素座標
            label = tk.Label(box, bg="#dddddd", bd=0, padx=0, pady=0, highlightthickness=0)
            label.pack(expand=True)
            self.__panels[key] = label

        # 只有左邊的原始影像可以作畫
        canvasLabel = self.__panels["original"]
        canvasLabel.bind("<ButtonPress-1>", self.onPress)
        canvasLabel.bind("<B1-Motion>", self.onDrag)
        canvasLabel.bind("<ButtonRelease-1>", self.onRelease)
        canvasLabel.bind("<Motion>", self.onMotion)
        canvasLabel.bind("<Leave>", self.onLeave)

        self.__status = tk.Label(root, text="請先選擇一張彩色影像，或選擇多張圖製作 GIF",
                                 anchor="w", padx=10, pady=8)
        self.__status.pack(fill="x")

    # ----- 原本的功能 -----

    def onSelectImage(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇彩色影像",
            filetypes=[("影像檔", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有檔案", "*.*")])
        if not path:
            return
        try:
            image = self.__fileManager.load(path)
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟影像：\n{e}")
            return

        self.__setOriginal(image, path)
        if not self.__gifFrames:          # 右邊在播 GIF 的話就讓它繼續播
            self.__convertedImage = None
            self.__panels["converted"].config(image="")
        self.__updateButtons()
        self.__status.config(text=f"已載入：{os.path.basename(path)}（選了筆觸就可以在左邊作畫）")

    def onConvert(self, mode: str) -> None:
        if self.__originalImage is None:
            return
        if mode == "grayscale":
            self.__converter = GrayscaleConverter()
        elif mode == "binary":
            self.__converter = BinaryConverter(threshold=128)
        elif mode == "palette":
            self.__converter = PaletteConverter(colorCount=256)
        elif mode == "xray":
            self.__converter = XrayConverter()
        elif mode == "median":
            self.__converter = MedianFilterConverter(size=int(self.__medianSize.get()))
        elif mode in ("dct_dc", "dct_mid", "dct_high"):
            self.__converter = DCTConverter(band=mode.split("_")[1])
        elif mode in ("lap_edge", "lap_sharpen"):
            self.__converter = LaplacianConverter(direction=self.__lapDirection.get(),
                                                  output=mode.split("_")[1])
            mode = f"{mode}_{self.__lapDirection.get()}"   # 存檔名稱帶上遮罩方向
        else:
            return

        self.__status.config(text="處理中…")
        self.__root.update_idletasks()
        # 轉換的是左邊「原圖 + 筆跡」的樣子
        result = self.__converter.convert(self.__workingImage())
        self.__stopGif()                  # 右邊改顯示轉換結果，原本的 GIF 就不播了
        self.__convertedImage = result
        self.__showImage(result, "converted")
        self.__suffix = mode
        self.__updateButtons()
        self.__status.config(text="轉換完成，需要的話請按「儲存成新檔」")

    def onUseResult(self) -> None:
        """把右邊的結果放到左邊當原圖，方便連續處理。
        右邊是 GIF 時：把目前顯示的那一格放到左邊（GIF 照樣繼續播或保持停止）。"""
        if self.__gifFrames:
            index = self.__gifShown
            self.__setOriginal(self.__gifFrames[index].convert("RGB"), self.__gifPaths[index])
            self.__updateButtons()
            self.__status.config(
                text=f"已把第 {index + 1} 張（{os.path.basename(self.__originalPath)}）放到左邊，可以作畫或轉換")
            return
        if self.__convertedImage is None:
            return
        self.__setOriginal(self.__convertedImage.convert("RGB"), self.__originalPath)
        self.__convertedImage = None
        self.__panels["converted"].config(image="")
        self.__updateButtons()
        self.__status.config(text="已將結果設為原圖，可以繼續處理")

    def onSave(self) -> None:
        rightHasResult = bool(self.__gifFrames) or self.__convertedImage is not None
        if rightHasResult and self.__hasStrokes:
            choice = messagebox.askyesnocancel(
                "要儲存哪一張？",
                "是：儲存右邊的轉換結果" + ("（GIF 動畫）" if self.__gifFrames else "") +
                "\n否：儲存左邊的作畫結果")
            if choice is None:
                return
            saveRight = choice
        else:
            saveRight = rightHasResult

        if saveRight and self.__gifFrames:
            self.__saveGif()
        elif saveRight:
            self.__saveImage(self.__convertedImage, self.__suffix)
        elif self.__hasStrokes:
            self.__saveImage(self.__fullCanvas.composite(), "drawing")

    def __saveImage(self, img: Image.Image, suffix: str) -> None:
        base = os.path.splitext(os.path.basename(self.__originalPath))[0] or "image"
        path = filedialog.asksaveasfilename(
            title="儲存成新檔",
            initialfile=f"{base}_{suffix}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp")])
        if not path:
            return
        try:
            self.__fileManager.save(img, path)
        except Exception as e:
            messagebox.showerror("錯誤", f"儲存失敗：\n{e}")
            return
        self.__status.config(text=f"已儲存：{path}")

    # ----- 第四版新增：三種筆觸（只畫在左邊） -----

    def onPickColor(self) -> None:
        rgb, _ = colorchooser.askcolor(color=self.__toHex(self.__color), title="選擇筆的顏色")
        if rgb is None:
            return
        self.__color = tuple(int(v) for v in rgb)
        self.__colorSwatch.config(bg=self.__toHex(self.__color))

    def onSelectBrush(self) -> None:
        """選了筆觸就可以馬上在左邊畫；不會影響右邊的 GIF"""
        name = {"pen": "畫筆", "eraser": "橡皮擦", "marker": "麥克筆"}[self.__brushName.get()]
        self.__updateSystemCursor()
        if self.__originalImage is None:
            self.__status.config(text=f"已選擇{name}，請先選擇一張影像")
        else:
            self.__status.config(text=f"已選擇{name}，可以直接在左邊的原始影像上畫")

    def onPress(self, event) -> None:
        if self.__previewCanvas is None:
            return
        if self.__brushName.get() == "none":
            self.__status.config(text="請先選擇筆觸（畫筆／橡皮擦／麥克筆）")
            return
        self.__mousePos = (event.x, event.y)
        brush = self.__currentBrush()
        brush.begin(self.__previewCanvas)
        brush.begin(self.__fullCanvas)
        self.__lastPoint = (event.x, event.y)
        self.__drawSegment(self.__lastPoint, self.__lastPoint)   # 只點一下也會留下一點

    def onDrag(self, event) -> None:
        if self.__lastPoint is None:
            return
        point = (event.x, event.y)
        self.__mousePos = point
        self.__drawSegment(self.__lastPoint, point)
        self.__lastPoint = point

    def onRelease(self, event) -> None:
        if self.__lastPoint is None:
            return
        brush = self.__currentBrush()
        brush.end(self.__previewCanvas)
        brush.end(self.__fullCanvas)
        self.__lastPoint = None
        self.__hasStrokes = True
        self.__refreshCanvas()
        self.__updateButtons()
        self.__status.config(text="作畫中…可以按轉換按鈕處理畫好的圖，或按「儲存成新檔」")

    def onClearDrawing(self) -> None:
        if self.__fullCanvas is None:
            return
        self.__fullCanvas.clear()
        self.__previewCanvas.clear()
        self.__hasStrokes = False
        self.__refreshCanvas()
        self.__updateButtons()
        self.__status.config(text="已清除全部筆跡")

    def onMotion(self, event) -> None:
        """滑鼠在左邊畫布上移動（沒按下）：更新游標圖示的位置"""
        self.__mousePos = (event.x, event.y)
        if self.__canShowBrushCursor():
            self.__refreshCanvas()

    def onLeave(self, event) -> None:
        """滑鼠離開畫布：把游標圖示拿掉"""
        self.__mousePos = None
        if self.__previewCanvas is not None:
            self.__refreshCanvas()

    def __canShowBrushCursor(self) -> bool:
        return self.__brushName.get() != "none" and self.__previewCanvas is not None

    def __updateSystemCursor(self) -> None:
        """選了筆觸就把系統箭頭藏起來（改由畫布上的筆圖示當游標），沒選就恢復箭頭"""
        label = self.__panels["original"]
        if self.__brushName.get() == "none":
            label.config(cursor="")
            return
        try:
            label.config(cursor="none")
        except tk.TclError:                 # 少數環境不支援隱藏游標，就改用內建的鉛筆游標
            label.config(cursor="pencil")

    def __currentBrush(self) -> Brush:
        return self.__brushes[self.__brushName.get()]

    def __drawSegment(self, p0, p1) -> None:
        """同一筆畫同時畫在預覽畫布和原尺寸畫布上"""
        brush = self.__currentBrush()
        width = int(self.__brushSize.get())
        s = self.__previewScale
        brush.drawSegment(self.__previewCanvas, p0, p1, self.__color, width)
        brush.drawSegment(self.__fullCanvas,
                          (p0[0] * s, p0[1] * s), (p1[0] * s, p1[1] * s),
                          self.__color, max(1, round(width * s)))
        self.__refreshCanvas()

    def __setOriginal(self, image: Image.Image, path: str) -> None:
        """設定左邊的原始影像，並把它變成新的畫布（筆跡清空）"""
        self.__originalImage = image.convert("RGB")
        self.__originalPath = path
        thumb = self.__originalImage.copy()
        thumb.thumbnail(self.PREVIEW_SIZE)
        self.__fullCanvas = DrawingCanvas(self.__originalImage)
        self.__previewCanvas = DrawingCanvas(thumb)
        self.__previewScale = self.__originalImage.width / thumb.width
        self.__lastPoint = None
        self.__hasStrokes = False
        self.__refreshCanvas()

    def __workingImage(self) -> Image.Image:
        """左邊目前的樣子：原圖 + 筆跡"""
        if self.__hasStrokes:
            return self.__fullCanvas.composite()
        return self.__originalImage

    def __refreshCanvas(self) -> None:
        img = self.__previewCanvas.composite()
        if self.__mousePos is not None and self.__canShowBrushCursor():
            img = img.convert("RGBA")
            self.__brushCursor.drawOn(img, self.__brushName.get(), self.__color, self.__mousePos)
        photo = ImageTk.PhotoImage(img)
        self.__photos["original"] = photo
        self.__panels["original"].config(image=photo)

    def __updateButtons(self) -> None:
        """依目前狀態啟用／停用按鈕"""
        hasImage = self.__originalImage is not None
        for btn in self.__convertButtons:
            btn.config(state="normal" if hasImage else "disabled")
        rightHasResult = bool(self.__gifFrames) or self.__convertedImage is not None
        self.__saveButton.config(state="normal" if (rightHasResult or self.__hasStrokes) else "disabled")
        self.__chainButton.config(state="normal" if rightHasResult else "disabled")

    # ----- 第四版新增：GIF 動畫（只有停止／播放按鈕會讓它停下或繼續） -----

    def onMakeGif(self) -> None:
        paths = filedialog.askopenfilenames(
            title="選擇多張圖（可多選，依檔名順序排列）",
            filetypes=[("影像檔", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有檔案", "*.*")])
        if not paths:
            return
        paths = GifMaker.sortByFileName(paths)
        try:
            images = [self.__fileManager.load(p) for p in paths]
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟影像：\n{e}")
            return

        self.__status.config(text="製作 GIF 中…")
        self.__root.update_idletasks()
        self.__stopGif()
        self.__gifFrames = self.__gifMaker.makeFrames(images)
        self.__gifThumbs = []
        for frame in self.__gifFrames:
            thumb = frame.convert("RGB")
            thumb.thumbnail(self.PREVIEW_SIZE)
            self.__gifThumbs.append(ImageTk.PhotoImage(thumb))
        self.__gifPaths = list(paths)
        self.__gifIndex = 0
        self.__gifShown = 0
        self.__convertedImage = None
        # 左邊顯示第一張圖，右邊播放動畫
        self.__setOriginal(images[0], paths[0])
        self.__gifCount.config(text=f"已選 {len(paths)} 張")
        self.__playButton.config(state="normal", text="⏸ 停止")
        self.__updateButtons()
        self.__status.config(text=f"已用 {len(paths)} 張圖做成 GIF（依檔名順序），按「儲存成新檔」存成 .gif")
        self.__animate()

    def onTogglePlay(self) -> None:
        """停止／繼續播放 GIF"""
        if not self.__gifFrames:
            return
        if self.__animJob is not None:   # 播放中 → 停止
            self.__root.after_cancel(self.__animJob)
            self.__animJob = None
            self.__playButton.config(text="▶ 播放")
            self.__status.config(
                text=f"GIF 停在第 {self.__gifShown + 1} / {len(self.__gifFrames)} 張，"
                     f"可按「以結果作為原圖」把這一格放到左邊")
        else:                            # 停止中 → 繼續播放
            self.__playButton.config(text="⏸ 停止")
            self.__status.config(text="播放 GIF 中…")
            self.__animate()

    def __saveGif(self) -> None:
        path = filedialog.asksaveasfilename(
            title="儲存 GIF 動畫",
            initialfile="animation.gif",
            defaultextension=".gif",
            filetypes=[("GIF 動畫", "*.gif")])
        if not path:
            return
        try:
            self.__gifMaker.save(self.__gifFrames, path, self.__durationMs())
        except Exception as e:
            messagebox.showerror("錯誤", f"儲存失敗：\n{e}")
            return
        self.__status.config(text=f"已儲存 GIF 動畫：{path}")

    def __animate(self) -> None:
        """在右邊預覽區輪流顯示每一格，模擬 GIF 播放"""
        if not self.__gifThumbs:
            return
        self.__panels["converted"].config(image=self.__gifThumbs[self.__gifIndex])
        self.__gifShown = self.__gifIndex
        self.__gifIndex = (self.__gifIndex + 1) % len(self.__gifThumbs)
        self.__animJob = self.__root.after(self.__durationMs(), self.__animate)

    def __stopGif(self) -> None:
        """清掉右邊的 GIF（換成轉換結果、或重新做 GIF 時用）"""
        if self.__animJob is not None:
            self.__root.after_cancel(self.__animJob)
            self.__animJob = None
        self.__gifFrames = None
        self.__gifThumbs = []
        self.__playButton.config(state="disabled", text="⏸ 停止")

    def __durationMs(self) -> int:
        try:
            return max(20, int(self.__gifDuration.get()))
        except ValueError:
            return 500

    # ----- 共用 -----

    @staticmethod
    def __toHex(rgb) -> str:
        return "#%02x%02x%02x" % tuple(rgb)

    def __showImage(self, img: Image.Image, panel: str = "converted") -> None:
        thumb = img.copy()
        thumb.thumbnail(self.PREVIEW_SIZE)
        photo = ImageTk.PhotoImage(thumb)
        self.__photos[panel] = photo
        self.__panels[panel].config(image=photo)


if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()
