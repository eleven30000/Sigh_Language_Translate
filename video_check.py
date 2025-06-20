#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檔名: video_validator.py
功能: 檢查指定路徑下所有 .mp4 是否可被 ffmpeg 完整解碼。
用法:
    python video_validator.py /path/to/videos --workers 8 --timeout 45
    python video_validator.py /path/to/a.mp4 /path/to/b.mp4
輸出:
    1. 在終端列印 [OK] / [CORRUPT]。
    2. 產生 corrupt_list.txt（只列壞檔），成功掃完則回傳 0；若 ffmpeg 不存在回傳 127。
"""

import argparse, subprocess, sys, textwrap, time
from pathlib import Path
from multiprocessing.pool import ThreadPool

# --------------------------- 參數解析 --------------------------- #
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        檢查影片完整性：
        - 預設遞迴掃描資料夾內所有 .mp4
        - 若指定多個 path，混用資料夾 / 檔案皆可
        - workers = 并行 ffmpeg 進程數
        - timeout = 單支影片最長允許秒數
    """)
)
parser.add_argument("paths", nargs="+", help="影片檔或資料夾")
parser.add_argument("--workers", type=int, default=4, help="並行數 (預設 4)")
parser.add_argument("--timeout", type=int, default=60, help="每支影片 timeout 秒數")
args = parser.parse_args()

# --------------------------- 檢查 ffmpeg --------------------------- #
def _check_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.stderr.write("❌ 找不到 ffmpeg，請先安裝再執行。\n")
        sys.exit(127)
_check_ffmpeg_installed()

# --------------------------- 收集影片清單 --------------------------- #
def gather_mp4(path: Path):
    if path.is_dir():
        yield from path.rglob("*.mp4")
    elif path.suffix.lower() == ".mp4":
        yield path
    else:
        sys.stderr.write(f"⚠️  跳過非 mp4：{path}\n")

video_files = []
for p in args.paths:
    video_files.extend(gather_mp4(Path(p)))

if not video_files:
    sys.stderr.write("⚠️  未找到任何 .mp4\n")
    sys.exit(1)

# --------------------------- 檢查單支影片 --------------------------- #
def validate_one(fp: Path):
    """返回 (fp, ok:bool, msg:str)"""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-nostdin", "-i", str(fp), "-f", "null", "-"]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.timeout
        )
        if res.returncode == 0 and res.stderr.strip() == "":
            return fp, True, ""
        else:
            return fp, False, res.stderr.strip()
    except subprocess.TimeoutExpired:
        return fp, False, f"timeout>{args.timeout}s"

# --------------------------- 多執行緒掃描 --------------------------- #
start = time.time()
corrupt = []
with ThreadPool(args.workers) as pool:
    for fp, ok, msg in pool.imap_unordered(validate_one, video_files):
        status = "OK" if ok else "CORRUPT"
        print(f"[{status}] {fp}")
        if not ok:
            corrupt.append((fp, msg))

# --------------------------- 結果輸出 --------------------------- #
print(f"\nDone. {len(video_files)-len(corrupt)}/{len(video_files)} passed. "
      f"耗時 {time.time()-start:.1f}s")

if corrupt:
    with open("corrupt_list.txt", "w", encoding="utf-8") as f:
        for fp, msg in corrupt:
            f.write(f"{fp}\t{msg}\n")
    print(f"⚠️  共 {len(corrupt)} 支影片異常，已寫入 corrupt_list.txt")
    sys.exit(2)
else:
    print("🎉 所有影片皆可正常解碼")
    sys.exit(0)
