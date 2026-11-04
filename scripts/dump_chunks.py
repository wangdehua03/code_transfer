#!/usr/bin/env python3
"""重新解码 手机录屏20261103.mp4，把每个唯一块实时写入 chunks_2/ 目录。

与 decode_from_video.py 的区别：块内容落盘为 chunk_XXX.txt，
便于与手机扫描的 0 号块合并后走 txt 目录模式重建。
其余解码逻辑（PhoneScanQRDecoder、区域定位、多尺度）完全复用原脚本。
"""

import os
import sys
from pathlib import Path

import decode_from_video as dvm

VIDEO = '手机录屏20261103.mp4'
OUT_DIR = Path('chunks_2')
OUT_DIR.mkdir(exist_ok=True)


def main():
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoder = dvm.PhoneScanQRDecoder()

    chunks = {}
    total_chunks = None
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % 50 == 0:
            print(f'帧 {fi}/{total} | 已存 {len(chunks)} 块', flush=True)
        for text in decoder.decode_frame(frame):
            index, chunk_total, data = dvm.parse_chunk_line(text)
            if index is not None:
                if total_chunks is None:
                    total_chunks = chunk_total
                if index not in chunks:
                    chunks[index] = text
                    (OUT_DIR / f'chunk_{index:03d}.txt').write_text(
                        text, encoding='utf-8')
                    print(f'  块 {index}/{chunk_total} 已保存', flush=True)
        fi += 1
    cap.release()
    print(f'\n完成: {len(chunks)}/{total_chunks} 块，缺失: '
          f'{sorted(set(range(total_chunks)) - set(chunks)) if total_chunks else "?"}')


if __name__ == '__main__':
    main()
