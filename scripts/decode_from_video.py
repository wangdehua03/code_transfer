#!/usr/bin/env python3
"""
从 QR 码视频或手机扫码 .txt 文件中还原数据。

支持两种输入模式:
  1. 视频文件（.mp4 等）— 逐帧检测 QR 码
     - PyBoof 手机扫码式解码（适用于手机录屏场景）
  2. .txt 文件目录 — 手机逐帧扫码场景（向后兼容）

流程:
  1. 编码端生成 qr_video.mp4（每帧一个 QR 码，内容为 "X/N,base64数据"）
  2. 本脚本从视频中逐帧提取 QR 码，或从 .txt 目录读取
  3. 按序号去重、排序、拼接、Base64 解码、解压 tar.gz

用法:
    python decode_from_video.py <video.mp4> [-o <输出目录>]
    python decode_from_video.py <txt文件目录> [-o <输出目录>]

示例:
    python decode_from_video.py ./qr_video.mp4
    python decode_from_video.py ./qr_video.mp4 -o ./restored/
    python decode_from_video.py ./scanned_qrs/
    python decode_from_video.py ./scanned_qrs/ -o ./restored/

要求:
    - 视频模式: 需要 OpenCV (cv2) 用于视频读取和图像预处理；需要 PyBoof (pyboof) 用于 QR 解码
    - txt 模式: 目录下放多个 .txt 文件（文件名任意）
    - 每个 QR 码 / .txt 内容格式: "X/N,base64编码数据"
      其中 X 是该块序号（0 到 N-1），N 是总块数
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import tarfile
from pathlib import Path

# 块头格式正则: "X/N,data" 或 "X/N;data"
CHUNK_PATTERN = re.compile(r'^(\d+)/(\d+)[,;]\s*(.*)', re.DOTALL)

# ---------------------------------------------------------------------------
# 依赖导入（模块级，供视频解码与 PyBoof 手机扫码式解码共用）
# ---------------------------------------------------------------------------
# OpenCV: 视频读取、图像预处理、QR 区域定位（不用于 QR 解码）
try:
    import cv2
except ImportError:
    cv2 = None

# PyBoof: BoofCV 的 Python 绑定，通过 py4j 启动 Java 进程。
# 它对模糊、透视畸变、小模块的高密度二维码（Version 40）解码能力远强于
# OpenCV 的 QRCodeDetector，是手机录屏场景的关键（唯一解码器）。
#
# 注意: PyBoof 默认使用固定端口 25333/25334 启动 Java 进程。若之前有残留的
# Java 进程占用这些端口（例如上次运行异常退出），会导致 "Address already in use"
# 错误。因此这里在 import 前自动挑选空闲端口，避免端口冲突。
def _pick_free_port(start=25333, count=50):
    """从 start 开始找 count 个空闲 TCP 端口，返回 (java_port, python_port)。"""
    import socket
    java_port = None
    python_port = None
    for port in range(start, start + count):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('127.0.0.1', port))
            if java_port is None:
                java_port = port
            else:
                python_port = port
                s.close()
                break
        except OSError:
            pass
        finally:
            s.close()
    if java_port is None or python_port is None:
        # 找不到足够空闲端口，回退到默认值（让 pyboof 自己报错）
        return 25333, 25334
    return java_port, python_port


try:
    import numpy as np
    # 在 import pyboof 前设置端口环境变量（pyboof 在 import 时读取）
    _java_port, _python_port = _pick_free_port()
    os.environ['PYBOOF_JAVA_PORT'] = str(_java_port)
    os.environ['PYBOOF_PYTHON_PORT'] = str(_python_port)
    import pyboof as pb
    PYBOOF_AVAILABLE = True
except ImportError:
    np = None
    pb = None
    PYBOOF_AVAILABLE = False

# Py4J 连接异常（Java 进程崩溃时抛出），用于检测连接断开并自动重启。
# pyboof 依赖 py4j，若 pyboof 可用则 py4j 必然可用；这里仍做防御性处理，
# 避免 py4j 缺失时整个脚本 import 失败。
try:
    from py4j.protocol import Py4JError, Py4JNetworkError
except ImportError:
    Py4JError = Exception
    Py4JNetworkError = Exception


class PhoneScanQRDecoder:
    """手机扫码式 QR 解码器（基于 PyBoof / BoofCV）。

    原理:
      手机录屏中二维码只占画面中央一部分（约 665x705 像素），且有透视畸变、
      H.264 压缩模糊。Version 40 高密度二维码模块仅约 3.6px/模块，
      OpenCV 的 QRCodeDetector 解码能力不足（几乎全部失败）。

      BoofCV 的二维码检测器对模糊、畸变、小模块的鲁棒性更好，因此:
        1. 定位 QR 区域：QR 背景为白色，找画面中最大的白色连通区域
        2. 裁剪出 QR 区域（去除周围无关内容，减少干扰）
        3. 多尺度放大（2x/2.5x/3x/4x），放大后模块更清晰、更易解码
        4. 用 PyBoof 解码，成功即停止

      性能优化:
        - 检测器全局创建一次，复用（PyBoof 通过 py4j 启动 Java 进程，首次较慢）
        - 上一帧定位到的区域加边距后优先尝试，失败再全帧重新定位
        - 多尺度从 2x 开始，成功即停止，避免不必要的计算

      健壮性（Java 进程崩溃容错）:
        PyBoof 通过 py4j 启动 Java 进程。长时间运行（>10 分钟）时，Java 进程
        可能因内存/资源问题崩溃，导致 Py4J 连接断开（Py4JNetworkError /
        ConnectionRefusedError / OSError 等）。此时若不做处理，整个解码会失败。

        本类在检测到连接断开时自动重启 Java 进程:
          - 捕获 Py4J 连接异常（Py4JNetworkError、Py4JError、OSError 等）
          - 打印警告，然后重启 PyBoof（关闭旧连接 → 重新 init → 重建检测器）
          - 重启后 last_region 保留（区域定位不依赖 Java 进程）
          - 限制重启次数（MAX_RESTARTS = 3），超过则放弃该帧继续下一帧
    """

    # 定位 QR 区域时要求的最小白色区域面积（避免误选 UI 小元素）
    MIN_REGION_AREA = 10000
    # 多尺度放大倍数（从 2x 开始，成功即停止）
    SCALES = [2.0, 2.5, 3.0, 4.0]
    # 复用上一帧区域时加的边距（像素）
    REGION_MARGIN = 20
    # Java 进程崩溃后允许的最大重启次数（超过则放弃该帧继续下一帧）
    MAX_RESTARTS = 3

    def __init__(self):
        # 全局复用检测器（不要每帧创建）
        self.detector = pb.FactoryFiducial(np.uint8).qrcode()
        self.last_region = None  # 上一帧定位到的区域 (x, y, w, h)
        self.restart_count = 0   # 已重启次数（用于限制重启次数）

    def _is_connection_error(self, exc):
        """判断异常是否为 Py4J 连接断开类异常（Java 进程崩溃）。

        覆盖 Py4JNetworkError、Py4JError（Java 侧无响应/对象不存在）、
        ConnectionRefusedError、OSError 等。这些异常都表明 Java 进程
        已不可用，需要重启。
        """
        if isinstance(exc, (Py4JNetworkError, Py4JError)):
            return True
        if isinstance(exc, (ConnectionRefusedError, OSError)):
            return True
        return False

    def _restart_pyboof(self):
        """重启 PyBoof Java 进程并重建检测器。

        流程:
          1. 尝试关闭旧连接（pb.shutdown_jvm()，内部会 shutdown gateway 并 kill Java 进程）
          2. 挑选新的空闲端口（避免旧端口被残留进程占用）
          3. 重新 pb.init_pyboof() 启动新 Java 进程
          4. 重新创建检测器 pb.FactoryFiducial(np.uint8).qrcode()

        注意:
          - last_region 不在此处重置（区域定位不依赖 Java 进程，重启后仍可复用）
          - 若重启失败（如 Java 无法启动），抛出异常由调用方处理
        """
        print("  [重启] PyBoof Java 进程连接断开，正在重启...")
        # 1. 关闭旧连接（可能已断开，需容错）
        try:
            pb.shutdown_jvm()
        except Exception as e:
            print(f"  [重启] 关闭旧连接时异常（可忽略）: {type(e).__name__}: {e}")
        # 2. 挑选新的空闲端口，避免与残留进程冲突
        java_port, python_port = _pick_free_port()
        print(f"  [重启] 使用新端口 java={java_port}, python={python_port}")
        # 3. 重新初始化 PyBoof（启动新 Java 进程）
        pb.init_pyboof(java_port=java_port, python_port=python_port)
        # 4. 重建检测器
        self.detector = pb.FactoryFiducial(np.uint8).qrcode()
        self.restart_count += 1
        print(f"  [重启] 完成（第 {self.restart_count} 次重启），检测器已重建")

    def _crop_and_decode(self, gray, region):
        """对指定区域裁剪、多尺度放大并用 PyBoof 解码。

        返回解码文本（已通过块格式校验）或 None。

        健壮性: 若 PyBoof Java 进程崩溃（连接断开），捕获异常并自动重启，
        重启后重试当前尺度；超过 MAX_RESTARTS 次则放弃该帧返回 None。
        """
        x, y, w, h = region
        if w < 50 or h < 50:
            return None
        # 必须 ascontiguousarray，PyBoof 需要连续内存
        crop = np.ascontiguousarray(gray[y:y + h, x:x + w])
        for scale in self.SCALES:
            # 每个尺度最多尝试 2 次: 第 0 次正常解码，若连接断开则重启后
            # 第 1 次重试当前尺度（避免重启后直接跳到下一尺度浪费一次机会）
            for attempt in range(2):
                try:
                    up = np.ascontiguousarray(cv2.resize(
                        crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC))
                    pb_img = pb.ndarray_to_boof(up)
                    self.detector.detect(pb_img)
                    if len(self.detector.detections) > 0:
                        msg = self.detector.detections[0].message
                        # 校验格式，避免误检污染 last_region
                        index, chunk_total, data = parse_chunk_line(msg)
                        if index is not None:
                            return msg
                    # 解码成功（无异常），跳出 attempt 循环，尝试下一尺度
                    break
                except Exception as e:
                    if self._is_connection_error(e):
                        # 检测到 Py4J 连接断开（Java 进程崩溃）→ 自动重启
                        if self.restart_count >= self.MAX_RESTARTS:
                            print(f"  [重启] 已超过最大重启次数 "
                                  f"({self.MAX_RESTARTS})，放弃当前帧")
                            return None
                        print(f"  [重启] 捕获连接异常: "
                              f"{type(e).__name__}: {e}")
                        try:
                            self._restart_pyboof()
                        except Exception as re:
                            # 重启失败（如 Java 无法启动），放弃当前帧
                            print(f"  [重启] 重启失败: "
                                  f"{type(re).__name__}: {re}")
                            return None
                        if attempt == 1:
                            # 重启后重试仍连接异常，放弃当前帧
                            print("  [重启] 重启后重试仍连接异常，放弃当前帧")
                            return None
                        # 重启成功，attempt 循环进入第 1 次重试当前尺度
                    else:
                        # 非连接类异常（如图像格式问题），忽略并继续下一尺度
                        break
        return None

    def locate_qr_region(self, gray):
        """定位 QR 区域：找画面中最大的白色连通区域。

        返回 (x, y, w, h) 或 None（未找到足够大的白色区域）。
        """
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        max_area = 0
        max_idx = -1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_idx = i
        if max_idx < 0 or max_area < self.MIN_REGION_AREA:
            return None
        x = stats[max_idx, cv2.CC_STAT_LEFT]
        y = stats[max_idx, cv2.CC_STAT_TOP]
        w = stats[max_idx, cv2.CC_STAT_WIDTH]
        h = stats[max_idx, cv2.CC_STAT_HEIGHT]
        return (x, y, w, h)

    def decode_frame(self, frame):
        """对一帧进行手机扫码式解码，返回解码文本列表（0 或 1 个）。

        优先用上一帧的区域（加边距）尝试，失败再全帧重新定位。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape

        # 候选区域列表：上一帧区域（加边距）优先，然后全帧重新定位
        regions = []
        if self.last_region is not None:
            x, y, w, h = self.last_region
            m = self.REGION_MARGIN
            x0 = max(0, x - m)
            y0 = max(0, y - m)
            x1 = min(fw, x + w + m)
            y1 = min(fh, y + h + m)
            regions.append((x0, y0, x1 - x0, y1 - y0))
        regions.append(self.locate_qr_region(gray))

        for region in regions:
            if region is None:
                continue
            msg = self._crop_and_decode(gray, region)
            if msg is not None:
                self.last_region = region
                return [msg]
        return []


# ---------------------------------------------------------------------------
# 通用: 解析与重建
# ---------------------------------------------------------------------------

def parse_chunk_line(line):
    """
    解析一行数据，格式: "X/N,base64数据" 或 "X/N;base64数据"
    返回 (序号, 总数, base64数据) 或 (None, None, None)
    """
    line = line.strip()
    if not line:
        return None, None, None

    match = CHUNK_PATTERN.match(line)
    if match:
        index = int(match.group(1))
        total = int(match.group(2))
        data = match.group(3).strip()
        return index, total, data
    else:
        return None, None, None


def load_from_directory(dir_path):
    """
    从目录中加载所有 .txt 文件（向后兼容：手机逐帧扫码场景）。
    不依赖文件名，直接从每个文件内容解析序号。
    文件名可以任意混乱，只要内容有序号前缀即可。
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        print(f"错误: 目录不存在: {dir_path}")
        sys.exit(1)

    files = list(dir_path.glob('*.txt'))
    if not files:
        print(f"错误: 在 {dir_path} 中未找到 .txt 文件")
        sys.exit(1)

    print(f"扫描目录: {dir_path}")
    print(f"找到 {len(files)} 个 .txt 文件\n")

    chunks = {}
    total = None
    skipped = 0

    for filepath in sorted(files):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read().strip()

        index, chunk_total, data = parse_chunk_line(content)
        if index is not None:
            chunks[index] = data
            if total is None:
                total = chunk_total
            print(f"  {filepath.name}: 块 {index}/{chunk_total}, 数据 {len(data)} 字节")
        else:
            skipped += 1
            print(f"  警告: {filepath.name} 未找到序号前缀，跳过")

    if not chunks:
        print("错误: 所有文件均未解析出序号")
        sys.exit(1)

    if total is not None:
        print(f"\n总块数 N = {total}")

    if skipped:
        print(f"跳过 {skipped} 个无效文件")

    return chunks, total


def load_index_json(video_path):
    """
    如果视频同目录下存在 index.json，读取并返回其内容。
    用于校验 archive_md5 和 total_chunks。
    """
    index_path = Path(video_path).parent / "index.json"
    if not index_path.is_file():
        return None

    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        print(f"已读取 index.json: {index_path}")
        print(f"  预期总块数: {index.get('total_chunks', '?')}")
        print(f"  预期 MD5:   {index.get('archive_md5', '?')}")
        return index
    except (json.JSONDecodeError, OSError) as e:
        print(f"警告: 读取 index.json 失败: {e}")
        return None


def reconstruct_and_decode(chunks, total, output_dir, expected_md5=None):
    """
    将分块数据按序号拼接，base64 解码，解压 tar.gz。

    参数:
        chunks: dict {序号: base64数据}
        total: 总块数
        output_dir: 输出目录
        expected_md5: 可选，index.json 中的 archive_md5 用于校验
    """
    if not chunks:
        print("错误: 未找到任何数据块")
        sys.exit(1)

    # 检查完整性
    expected_keys = set(range(total))
    actual_keys = set(chunks.keys())
    missing = expected_keys - actual_keys

    if missing:
        print(f"\n找到 {len(actual_keys)}/{total} 个数据块（不完整）")
        print(f"缺失的块序号: {sorted(missing)}")
        sys.exit(1)

    print(f"\n找到 {len(actual_keys)}/{total} 个数据块（完整）")

    # 按序号排序并拼接
    sorted_keys = sorted(chunks.keys())
    b64_combined = ""
    for key in sorted_keys:
        data = chunks[key]
        b64_combined += data
        print(f"  块 {key:>3}: {len(data):>6} 字节")

    print(f"\n总 base64 长度: {len(b64_combined):,} 字节")

    # Base64 解码
    print("正在 Base64 解码...")
    try:
        decoded = base64.b64decode(b64_combined)
    except Exception as e:
        print(f"错误: Base64 解码失败: {e}")
        print("提示: 请检查数据中是否混入了非 base64 字符（如空格、换行）")
        sys.exit(1)

    print(f"解码后大小: {len(decoded):,} 字节 ({len(decoded) / 1024:.1f} KB)")

    # 计算校验和
    md5 = hashlib.md5(decoded).hexdigest()
    sha256 = hashlib.sha256(decoded).hexdigest()
    print(f"MD5:    {md5}")
    print(f"SHA256: {sha256}")

    # 如果提供了预期 MD5，进行校验
    if expected_md5:
        if md5 == expected_md5:
            print(f"✅ MD5 校验通过（与 index.json 一致）")
        else:
            print(f"❌ MD5 校验失败！")
            print(f"   预期: {expected_md5}")
            print(f"   实际: {md5}")
            print("   数据可能损坏，请重新录制视频或检查编码过程")
            sys.exit(1)

    # 写入 tar.gz
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "archive.tar.gz"
    with open(archive_path, 'wb') as f:
        f.write(decoded)
        f.flush()
        os.fsync(f.fileno())
    print(f"\n已写入: {archive_path} ({os.path.getsize(archive_path)} 字节)")

    # 解压
    print("正在解压...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getnames()
        print(f"压缩包包含 {len(members)} 个文件/目录:")
        for m in members[:20]:
            print(f"  {m}")
        if len(members) > 20:
            print(f"  ... 还有 {len(members) - 20} 个")
        tar.extractall(path=out)

    # 显示解压结果
    extracted = list(out.iterdir())
    extracted_dirs = [e for e in extracted if e.is_dir()]
    print(f"\n{'=' * 60}")
    print("还原完成！")
    print(f"输出目录: {out.resolve()}")
    print(f"解压出 {len(extracted_dirs)} 个目录:")
    for d in extracted_dirs:
        file_count = sum(1 for _ in d.rglob('*') if _.is_file())
        print(f"  {d.name}/  ({file_count} 个文件)")
    print(f"{'=' * 60}")

    return out


# ---------------------------------------------------------------------------
# 视频解码
# ---------------------------------------------------------------------------

def decode_from_video(video_path, output_dir):
    """
    从视频中逐帧检测 QR 码，提取分块数据，拼接解码并解压。

    参数:
        video_path: 视频文件路径（.mp4 等）
        output_dir: 输出目录

    解码策略:
        PyBoof 手机扫码式解码（定位QR区域→裁剪→多尺度放大→PyBoof解码）
        适用于手机录屏场景（透视畸变、压缩模糊、小模块）
    """
    if cv2 is None:
        print("错误: 未安装 OpenCV (cv2)，无法读取视频")
        print("安装方法: pip install opencv-python")
        sys.exit(1)

    video_path = Path(video_path)
    if not video_path.is_file():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    print(f"视频文件: {video_path}")
    print(f"文件大小: {video_path.stat().st_size:,} 字节\n")

    # 尝试读取 index.json 用于校验
    index = load_index_json(video_path)
    expected_md5 = index.get('archive_md5') if index else None
    expected_total = index.get('total_chunks') if index else None

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {width}x{height}, {fps:.1f} FPS, 共 {total_frames} 帧")
    print(f"预计时长: {total_frames / fps:.1f} 秒\n")

    # 创建 PyBoof 手机扫码式解码器（唯一解码器）
    if not PYBOOF_AVAILABLE:
        print("错误: 未安装 PyBoof (pyboof)，无法进行 QR 解码")
        print("安装方法: pip install pyboof")
        sys.exit(1)
    try:
        decoder = PhoneScanQRDecoder()
        print("解码器: PyBoof 手机扫码式（定位QR区域→裁剪→多尺度放大→解码）")
    except Exception as e:
        print(f"错误: PyBoof 初始化失败: {e}")
        print("请检查 Java 环境及 pyboof 安装是否正确")
        sys.exit(1)

    chunks = {}
    total = None
    decoded_count = 0
    frame_idx = 0
    # 过渡帧提示标记: 手机录屏视频开头前 ~40 帧是过渡帧（黑屏/过渡动画），
    # 没有 QR 码。若前 100 帧仍未解出任何块，打印提示但继续扫描。
    transition_hint_printed = False

    print("开始逐帧检测 QR 码...")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每 10 帧打印一次进度
        if frame_idx % 10 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  帧 {frame_idx:>5}/{total_frames} ({pct:5.1f}%) | "
                  f"已解码 {len(chunks)} 块"
                  + (f" / {total}" if total else ""))

        # 解码当前帧（PyBoof 手机扫码式）
        texts = decoder.decode_frame(frame)

        for text in texts:
            index, chunk_total, data = parse_chunk_line(text)
            if index is not None:
                if total is None:
                    total = chunk_total
                # 去重：只保留第一次成功解码的
                if index not in chunks:
                    chunks[index] = data
                    decoded_count += 1

        frame_idx += 1

        # 过渡帧提示: 前 100 帧仍未解出任何块时，提示可能是过渡帧，
        # 但不要提前退出，继续扫描（用户可能录错了，整个视频都是过渡帧）。
        if (not transition_hint_printed and frame_idx >= 100
                and len(chunks) == 0):
            print("  提示: 前 100 帧未检测到任何 QR 码，"
                  "视频开头可能是过渡帧（黑屏/过渡动画），继续扫描...")
            transition_hint_printed = True

    cap.release()

    print("-" * 60)
    print(f"视频扫描完成: 共 {frame_idx} 帧, 成功解码 {decoded_count} 个唯一块")

    if not chunks:
        print("错误: 未从视频中检测到任何有效 QR 码")
        print("提示: 请检查视频质量（清晰度、光线、稳定性）")
        sys.exit(1)

    # 确定总块数
    if total is None:
        total = max(chunks.keys()) + 1
        print(f"警告: 未能从 QR 码中确定总块数，使用最大值+1 = {total}")

    # 如果 index.json 提供了总块数，优先使用
    if expected_total and expected_total != total:
        print(f"警告: index.json 中总块数为 {expected_total}，"
              f"但 QR 码中检测到 {total}")
        # 以 QR 码中检测到的为准（可能 index.json 不匹配）
        total = max(total, expected_total)

    print(f"总块数 N = {total}\n")

    # 拼接、解码、解压
    reconstruct_and_decode(chunks, total, output_dir, expected_md5=expected_md5)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='从 QR 码视频或手机扫码 .txt 文件中还原数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从视频直接解码（推荐）
  python decode_from_video.py ./qr_video.mp4
  python decode_from_video.py ./qr_video.mp4 -o ./restored/

  # 从手机扫码 .txt 目录解码（向后兼容）
  python decode_from_video.py ./scanned_qrs/
  python decode_from_video.py ./scanned_qrs/ -o ./restored/

说明:
  编码端生成 qr_video.mp4，每帧一个 QR 码，内容为 "X/N,base64数据"。
  本脚本支持两种输入:
    1. 视频文件 — PyBoof 手机扫码式解码（需安装 pyboof）
    2. .txt 目录 — 手机逐帧扫码得到的文本文件
  按序号排序、拼接、Base64 解码、解压 tar.gz 还原原始文件。
        """
    )
    parser.add_argument(
        'input',
        help='视频文件路径（.mp4 等）或包含扫码 .txt 文件的目录'
    )
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='输出目录（默认: output）'
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        # 向后兼容: 从 .txt 目录解码
        print("模式: 从 .txt 目录解码（手机扫码场景）\n")
        chunks, total = load_from_directory(input_path)
        if not chunks:
            print("错误: 未找到任何数据")
            sys.exit(1)
        reconstruct_and_decode(chunks, total, args.output)
    elif input_path.is_file():
        # 视频解码
        print("模式: 从视频解码\n")
        decode_from_video(input_path, args.output)
    else:
        print(f"错误: 不支持的输入类型: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
