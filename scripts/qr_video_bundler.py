#!/usr/bin/env python3
"""
QR Video Bundler — Pack files into a QR-code video.

Workflow:
  1. Pack input files/directory into archive.tar.gz
  2. Base64-encode the archive
  3. Split into chunks (each chunk ≤ 2900 chars of Base64 data)
  4. Generate a QR Code PNG for each chunk
  5. Synthesize all QR frames into an MP4 video via FFmpeg
  6. Clean up temporary frame images
  7. Write index.json with metadata
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import qrcode
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_DATA_PER_CHUNK = 2300  # safe payload for QR Version 40 + EC M (~15% tolerance)
QR_VERSION = 40
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_M
BOX_SIZE = 8  # px per module
BORDER = 4    # quiet zone modules (must be ≥4 for OpenCV QRCodeDetector)
FRAMES_PER_QR = 4
VIDEO_FPS = 5
ARCHIVE_NAME = "archive.tar.gz"
VIDEO_NAME = "qr_video.mp4"
INDEX_NAME = "index.json"
FRAME_PREFIX = "frame_"
FRAME_EXT = ".png"
FRAME_PATTERN = f"{FRAME_PREFIX}%04d{FRAME_EXT}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5_file(path: str) -> str:
    """Return hex MD5 digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_archive(inputs: list[str], archive_path: str) -> None:
    """Pack *inputs* (files and/or directories) into *archive_path* (tar.gz)."""
    print(f"[1/6] Packing {len(inputs)} input(s) into {ARCHIVE_NAME} ...")
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in inputs:
            item = os.path.abspath(item)
            if os.path.isfile(item):
                arcname = os.path.basename(item)
                tar.add(item, arcname=arcname)
            elif os.path.isdir(item):
                tar.add(item, arcname=os.path.basename(item))
            else:
                raise FileNotFoundError(f"Input not found: {item}")
    size = os.path.getsize(archive_path)
    print(f"       Archive size: {size:,} bytes")


def encode_and_chunk(archive_path: str) -> tuple[list[str], int]:
    """Base64-encode the archive and split into chunks.

    Returns (chunks, b64_length) where each chunk is like "X/N,data".
    """
    print("[2/6] Base64 encoding and chunking ...")
    with open(archive_path, "rb") as f:
        raw = f.read()
    b64_bytes = base64.b64encode(raw)
    b64_str = b64_bytes.decode("ascii")
    b64_length = len(b64_str)

    # Split into chunks of MAX_DATA_PER_CHUNK characters
    total_chunks = (b64_length + MAX_DATA_PER_CHUNK - 1) // MAX_DATA_PER_CHUNK
    chunks: list[str] = []
    for i in range(total_chunks):
        start = i * MAX_DATA_PER_CHUNK
        end = min(start + MAX_DATA_PER_CHUNK, b64_length)
        data_slice = b64_str[start:end]
        chunk_header = f"{i}/{total_chunks},{data_slice}"
        chunks.append(chunk_header)
    print(f"       Base64 length: {b64_length:,} chars")
    print(f"       Total chunks:  {total_chunks}")
    return chunks, b64_length


def generate_qr_with_mask_retry(data: str, version: int = QR_VERSION,
                                box_size: int = BOX_SIZE, border: int = BORDER) -> Image.Image:
    """Generate a QR code, auto-selecting a mask pattern that OpenCV can decode.

    Tries all 8 mask patterns (0-7) and returns the first one whose rendered
    image is successfully decoded by cv2.QRCodeDetector.
    """
    detector = cv2.QRCodeDetector()

    for mask in range(8):
        qr = qrcode.QRCode(
            version=version,
            error_correction=QR_ERROR_CORRECTION,
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.makeImpl(False, mask)  # force a specific mask pattern
        img: Image.Image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        arr = np.array(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        decoded, _, _ = detector.detectAndDecode(bgr)
        if decoded == data:  # verify decoded result matches original data
            return img

    raise RuntimeError(
        f"无法找到 OpenCV 能解码的 mask pattern for data: {data[:50]}..."
    )


def generate_qr_frames(chunks: list[str], frame_dir: str, frames_per_qr: int = FRAMES_PER_QR) -> int:
    """Generate QR Code PNG frames. Each QR is repeated *frames_per_qr* times.

    Returns the total number of frame images written.
    """
    print(f"[3/6] Generating QR Code frames ({frames_per_qr} copies per QR) ...")
    os.makedirs(frame_dir, exist_ok=True)
    total_frames = 0
    for idx, chunk in enumerate(chunks):
        img = generate_qr_with_mask_retry(chunk)

        # Save frames_per_qr copies
        for copy in range(frames_per_qr):
            frame_idx = idx * frames_per_qr + copy
            frame_name = f"{FRAME_PREFIX}{frame_idx:04d}{FRAME_EXT}"
            frame_path = os.path.join(frame_dir, frame_name)
            img.save(frame_path, "PNG")
            total_frames += 1

        if (idx + 1) % 10 == 0 or idx + 1 == len(chunks):
            print(f"       QR {idx + 1}/{len(chunks)} done")

    print(f"       Total frames written: {total_frames}")
    return total_frames


def synthesize_video(frame_dir: str, output_path: str, fps: int = VIDEO_FPS) -> None:
    """Use FFmpeg to synthesize all frame images into an MP4 video."""
    print(f"[4/6] Synthesizing video with FFmpeg (fps={fps}) ...")
    # Build input pattern: read from frame_dir
    input_pattern = os.path.join(frame_dir, FRAME_PATTERN)

    # Read actual frame dimensions to prevent FFmpeg from auto-scaling
    # (H.264 requires even dimensions; odd sizes get silently resized,
    #  which distorts QR codes and makes them undecodable)
    first_frame = os.path.join(frame_dir, f"{FRAME_PREFIX}0000{FRAME_EXT}")
    with Image.open(first_frame) as img:
        fw, fh = img.size
    print(f"       Frame size: {fw}x{fh}")

    cmd = [
        "ffmpeg",
        "-y",                       # overwrite output
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf", f"scale={fw}:{fh}",  # preserve exact frame size
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",       # suppress normal FFmpeg output
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"       FFmpeg stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")
    size = os.path.getsize(output_path)
    print(f"       Video size: {size:,} bytes")


def cleanup_frame_dir(frame_dir: str) -> None:
    """Remove the temporary frame directory."""
    print("[5/6] Cleaning up temporary frame images ...")
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
        print("       Done")


def write_index(output_dir: str, archive_path: str, b64_length: int,
                total_chunks: int, fps: int, frames_per_qr: int) -> None:
    """Write index.json with metadata."""
    print("[6/6] Writing index.json ...")
    archive_size = os.path.getsize(archive_path)
    archive_md5 = md5_file(archive_path)
    video_duration = (total_chunks * frames_per_qr) / fps

    index = {
        "version": "1.0",
        "tool": "qr_video_bundler",
        "archive_file": ARCHIVE_NAME,
        "archive_size": archive_size,
        "archive_md5": archive_md5,
        "b64_length": b64_length,
        "total_chunks": total_chunks,
        "max_data_per_chunk": MAX_DATA_PER_CHUNK,
        "qr_code_version": QR_VERSION,
        "error_correction": "M",
        "video_fps": fps,
        "frames_per_qr": frames_per_qr,
        "video_duration_sec": video_duration,
    }
    index_path = os.path.join(output_dir, INDEX_NAME)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"       index.json written to {index_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def bundle(inputs: list[str], output_dir: str) -> None:
    """Run the full bundling pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    archive_path = os.path.join(output_dir, ARCHIVE_NAME)
    video_path = os.path.join(output_dir, VIDEO_NAME)

    # Use a temporary directory for frame images
    frame_dir = tempfile.mkdtemp(prefix="qr_frames_")

    try:
        # Step 1: pack
        pack_archive(inputs, archive_path)

        # Step 2: encode & chunk
        chunks, b64_length = encode_and_chunk(archive_path)

        # Step 3: generate QR frames
        generate_qr_frames(chunks, frame_dir, FRAMES_PER_QR)

        # Step 4: synthesize video
        synthesize_video(frame_dir, video_path, VIDEO_FPS)

        # Step 5: cleanup
        cleanup_frame_dir(frame_dir)
        frame_dir = None  # prevent double-cleanup in except

        # Step 6: index
        write_index(output_dir, archive_path, b64_length, len(chunks),
                    VIDEO_FPS, FRAMES_PER_QR)

        print("\n✅ Bundling complete!")
        print(f"   Output directory: {output_dir}")
        print(f"   Files:")
        for name in [ARCHIVE_NAME, VIDEO_NAME, INDEX_NAME]:
            p = os.path.join(output_dir, name)
            if os.path.exists(p):
                print(f"     - {name} ({os.path.getsize(p):,} bytes)")
    finally:
        if frame_dir and os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack files into a QR-code video for transfer."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- bundle sub-command ---
    bundle_parser = sub.add_parser("bundle", help="Bundle files/directory into a QR video")
    bundle_parser.add_argument(
        "files", nargs="*", default=[],
        help="Input files to bundle"
    )
    bundle_parser.add_argument(
        "-d", "--directory", default=None,
        help="Input directory to bundle (alternative to listing files)"
    )
    bundle_parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    if args.command == "bundle":
        inputs: list[str] = []
        if args.directory:
            inputs.append(args.directory)
        if args.files:
            inputs.extend(args.files)
        if not inputs:
            parser.error("Provide at least one file or use -d <directory>")
        bundle(inputs, args.output)


if __name__ == "__main__":
    main()
