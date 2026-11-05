#!/usr/bin/env python3
"""重新解码视频，把每个唯一块实时写入块目录。

与 decode_from_video.py 的区别：块内容落盘为 chunk_XXX.txt，
便于与手机扫描的缺失块合并后走 txt 目录模式重建。
其余解码逻辑（PhoneScanQRDecoder、区域定位、多尺度）完全复用原脚本。

用法:
    python dump_chunks.py <视频文件> <块输出目录>
示例:
    python dump_chunks.py 手机录屏20261103.mp4 chunks_2
"""

import os
import sys
from pathlib import Path

import decode_from_video as dvm


def main():
    import cv2
    import numpy as np

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    video, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(exist_ok=True)
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
                    (out_dir / f'chunk_{index:03d}.txt').write_text(
                        text, encoding='utf-8')
                    print(f'  块 {index}/{chunk_total} 已保存', flush=True)
        fi += 1
    cap.release()
    print(f'\n完成: {len(chunks)}/{total_chunks} 块，缺失: '
          f'{sorted(set(range(total_chunks)) - set(chunks)) if total_chunks else "?"}')


if __name__ == '__main__':
    main()
