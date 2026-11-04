#!/usr/bin/env python3
"""
QR Image Bundler — Pack small files into multiple QR-code images.

Workflow:
  1. Pack input files/directory into archive.tar.gz
  2. Base64-encode the archive
  3. Split into chunks (each chunk ≤ 2300 chars of Base64 data)
  4. Generate one QR Code PNG image for each chunk
  5. Write index.json with metadata

This is a simplified version of qr_video_bundler: instead of synthesizing a
video, it outputs one static QR image per chunk. Users scan each image with
their phone to get the text content, then reassemble and decompress on the
external network. Suitable for small files (<100KB) crossing network
boundaries.
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import tarfile

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
ARCHIVE_NAME = "archive.tar.gz"
INDEX_NAME = "index.json"
QR_PREFIX = "qr_"
QR_EXT = ".png"


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
    print(f"[1/4] Packing {len(inputs)} input(s) into {ARCHIVE_NAME} ...")
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
    print("[2/4] Base64 encoding and chunking ...")
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


def generate_qr_images(chunks: list[str], output_dir: str) -> int:
    """Generate one QR Code PNG image per chunk.

    Images are named qr_0001.png, qr_0002.png, ... (1-based, zero-padded to 4
    digits). Returns the total number of images written.
    """
    print("[3/4] Generating QR Code images ...")
    os.makedirs(output_dir, exist_ok=True)
    total_images = 0
    for idx, chunk in enumerate(chunks):
        img = generate_qr_with_mask_retry(chunk)

        image_name = f"{QR_PREFIX}{idx + 1:04d}{QR_EXT}"
        image_path = os.path.join(output_dir, image_name)
        img.save(image_path, "PNG")
        total_images += 1

        if (idx + 1) % 10 == 0 or idx + 1 == len(chunks):
            print(f"       QR {idx + 1}/{len(chunks)} done")

    print(f"       Total images written: {total_images}")
    return total_images


def write_index(output_dir: str, archive_path: str, b64_length: int,
                total_chunks: int) -> None:
    """Write index.json with metadata."""
    print("[4/4] Writing index.json ...")
    archive_size = os.path.getsize(archive_path)
    archive_md5 = md5_file(archive_path)

    index = {
        "version": "1.0",
        "tool": "qr_image_bundler",
        "archive_file": ARCHIVE_NAME,
        "archive_size": archive_size,
        "archive_md5": archive_md5,
        "b64_length": b64_length,
        "total_chunks": total_chunks,
        "max_data_per_chunk": MAX_DATA_PER_CHUNK,
        "qr_code_version": QR_VERSION,
        "error_correction": "M",
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

    # Step 1: pack
    pack_archive(inputs, archive_path)

    # Step 2: encode & chunk
    chunks, b64_length = encode_and_chunk(archive_path)

    # Step 3: generate QR images
    total_images = generate_qr_images(chunks, output_dir)

    # Step 4: index
    write_index(output_dir, archive_path, b64_length, len(chunks))

    print("\n✅ Bundling complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Files:")
    for name in [ARCHIVE_NAME, INDEX_NAME]:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            print(f"     - {name} ({os.path.getsize(p):,} bytes)")
    if total_images > 0:
        first = f"{QR_PREFIX}0001{QR_EXT}"
        last = f"{QR_PREFIX}{total_images:04d}{QR_EXT}"
        if total_images == 1:
            print(f"     - {first} (1 image)")
        else:
            print(f"     - {first} ... {last} ({total_images} images)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack small files into multiple QR-code images for transfer."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- bundle sub-command ---
    bundle_parser = sub.add_parser("bundle", help="Bundle files/directory into QR images")
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
