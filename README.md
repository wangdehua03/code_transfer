# code_transfer

通过**二维码视频**在隔离网络（或任何只能"用眼睛看"的渠道）之间搬运文件的工具链。

原理：把任意文件打包成 tar.gz → Base64 分块 → 每块生成一个高密度二维码（Version 40）→ 串成视频在源端播放 → 目的端用手机录屏（或逐帧扫码）→ 本工具逐帧解码、按序号拼接还原。

## 工作流程

```
源端                                目的端
────                                ────
qr_video_bundler.py                 手机录屏
  (文件 → QR视频 qr_video.mp4)  ──>  decode_from_video.py
                                      (视频 → 还原文件)
                            或:  手机逐帧扫码存 .txt ──> decode_from_video.py <txt目录>
```

## 目录结构

```
scripts/
├── qr_video_bundler.py    # 视频编码端：文件/目录 → 分块 → QR 视频（需 qrcode、Pillow、FFmpeg）
├── decode_from_video.py   # 视频解码端：视频 或 扫码 .txt 目录 → 还原文件（需 cv2、pyboof）
├── dump_chunks.py         # 辅助：解码视频并把每个块落盘为 chunk_XXX.txt，
│                          # 便于用手机补扫缺失块后走 txt 目录模式重建
├── qr_image_bundler.py    # 图片编码端（简化版，适合 <100KB 小文件）：文件 → 每块一张 QR 图片
└── decode_from_images.py  # 图片解码端：QR 图片目录 或 扫码 .txt 目录 → 还原文件
```

## 安装

建议使用 conda 虚拟环境（Python 3.11）：

```bash
pip install -r requirements.txt
# 解码端还需要 Java 运行环境（pyboof 通过 py4j 启动 Java 进程）
```

版本经实测组合：**numpy 1.26.2 + opencv-python-headless 4.11.0.86 + pyboof 0.43.1**。
请勿升级 numpy 到 2.x / opencv 到 5.x，pyboof 0.43.1 与新版 numpy 的内存映射传图不兼容。

## 使用

### 源端编码

```bash
python scripts/qr_video_bundler.py -d <要传输的目录> -o <输出目录>
# 产出 qr_video.mp4 + index.json（含 MD5 校验值）
```

### 目的端解码（视频模式）

```bash
python scripts/decode_from_video.py <手机录屏.mp4> [-o 输出目录]
```

### 目的端解码（手机逐帧扫码模式，向后兼容）

```bash
python scripts/decode_from_video.py <txt文件目录> [-o 输出目录]
# 每个 txt 内容格式: "X/N,base64数据"，文件名任意
```

### 图片模式（小文件简化版）

源端生成每块一张的二维码图片（无需 FFmpeg，适合 <100KB 的小文件）：

```bash
python scripts/qr_image_bundler.py <文件或目录> -o <输出目录>
# 产出 qr_0001.png、qr_0002.png ... + index.json
```

目的端从图片目录或手机扫码 txt 目录还原：

```bash
python scripts/decode_from_images.py <图片目录或txt目录> [-o 输出目录]
```

### 缺块补救

视频模式若提示缺失块序号，可用手机对准源端播放器补扫缺失的块：

```bash
python scripts/dump_chunks.py        # 先把已解出的块落盘（修改脚本内 VIDEO 路径）
# 手机扫出的缺失块存为 chunk_000.txt 等，放入同目录
python scripts/decode_from_video.py <块目录> -o 输出目录
```

## 已知问题

### pyboof 0.43.1 在 Windows 下的 mmap bug（重要）

pyboof 用内存映射文件在 Python 与 Java 进程间传图。0.43.1 版本的
`pyboof/__init__.py` 中 `__init_memmap()` 的 Windows 分支把 mmap 对象赋给了
**局部变量**而非 `pbg.mmap_file`，导致 Windows 上图像映射从未生效：

- Python→Java 传图退化为异常的兜底路径，二维码检出率为 0
- Java→Python 读结果时必然抛 `AttributeError: 'NoneType' object has no attribute 'seek'`

Linux 无此问题。上游 master 已修复但未发版。

**修复方法**（二选一）：

1. 手动补丁：编辑 `site-packages/pyboof/__init__.py`，找到 `__init_memmap` 中

   ```python
   if os.name == 'nt':
       mmap_file = mmap.mmap(pbg.mmap_fid.fileno(), length=0)
   ```

   改为 `pbg.mmap_file = mmap.mmap(...)`（即加上 `pbg.` 前缀）。
   注意：重装/升级 pyboof 后需重新打补丁。

2. 从上游源码安装（需要 JDK + Gradle，会现场编译 BoofCV jar）。

## 实测参考

- Version 40（177×177 模块）高密度二维码，手机拍屏录屏场景，模块约 2.5~3.6 px：
  OpenCV QRCodeDetector 几乎全军覆没，BoofCV（pyboof）是唯一能解的引擎。
- 192 秒录屏（5753 帧 / 235 块）约 30 分钟解完。
