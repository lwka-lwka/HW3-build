"""影像轉換系統（HW3：HW2 + 中值濾波 + DCT 轉換）
使用者選擇一張彩色影像，轉換成灰階、純黑白、調色盤或 X 光影像，
也可以做中值濾波（去除胡椒鹽雜訊）以及 DCT 轉換後把 DC / 中頻 / 高頻係數歸 0，
轉換後可按「儲存成新檔」存檔。

需要套件：Pillow、NumPy  (pip install pillow numpy)
"""

from abc import ABC, abstractmethod
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageOps, ImageTk


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

        root.title("影像轉換系統")
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

        # 「以結果繼續處理」：例如對中值濾波的結果再濾一次
        self.__chainButton = tk.Button(bar2, text="以結果作為原圖", state="disabled",
                                       command=self.onUseResult)
        self.__chainButton.pack(side="right", padx=4)

        # 預覽區：左邊原圖、右邊轉換結果
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
            label = tk.Label(box, bg="#dddddd")
            label.pack(expand=True)
            self.__panels[key] = label

        self.__status = tk.Label(root, text="請先選擇一張彩色影像",
                                 anchor="w", padx=10, pady=8)
        self.__status.pack(fill="x")

    def onSelectImage(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇彩色影像",
            filetypes=[("影像檔", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有檔案", "*.*")])
        if not path:
            return
        try:
            self.__originalImage = self.__fileManager.load(path)
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟影像：\n{e}")
            return

        self.__originalPath = path
        self.__convertedImage = None
        self.__showImage(self.__originalImage, "original")
        self.__panels["converted"].config(image="")
        for btn in self.__convertButtons:
            btn.config(state="normal")
        self.__saveButton.config(state="disabled")
        self.__chainButton.config(state="disabled")
        self.__status.config(text=f"已載入：{os.path.basename(path)}")

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
        else:
            return

        self.__status.config(text="處理中…")
        self.__root.update_idletasks()
        self.__convertedImage = self.__converter.convert(self.__originalImage)
        self.__showImage(self.__convertedImage, "converted")
        self.__suffix = mode
        self.__saveButton.config(state="normal")
        self.__chainButton.config(state="normal")
        self.__status.config(text="轉換完成，需要的話請按「儲存成新檔」")

    def onUseResult(self) -> None:
        """把目前的轉換結果當成新的原圖，方便連續處理"""
        if self.__convertedImage is None:
            return
        self.__originalImage = self.__convertedImage.convert("RGB")
        self.__convertedImage = None
        self.__showImage(self.__originalImage, "original")
        self.__panels["converted"].config(image="")
        self.__saveButton.config(state="disabled")
        self.__chainButton.config(state="disabled")
        self.__status.config(text="已將轉換結果設為原圖，可以繼續處理")

    def onSave(self) -> None:
        if self.__convertedImage is None:
            return
        base = os.path.splitext(os.path.basename(self.__originalPath))[0]
        path = filedialog.asksaveasfilename(
            title="儲存成新檔",
            initialfile=f"{base}_{self.__suffix}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp")])
        if not path:
            return
        try:
            self.__fileManager.save(self.__convertedImage, path)
        except Exception as e:
            messagebox.showerror("錯誤", f"儲存失敗：\n{e}")
            return
        self.__status.config(text=f"已儲存：{path}")

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
