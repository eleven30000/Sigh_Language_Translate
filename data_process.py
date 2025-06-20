import os
import json
import copy
from typing import List, Dict, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
from decord import VideoReader, cpu

from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
device = "cuda"

# -----------------------------------------------------------------------------
# 1. Video loading function (UNCHANGED)
# -----------------------------------------------------------------------------

def load_video(path, T=16):
    try:
        vr = VideoReader(path, ctx=cpu(0))
        print (vr)
        idx = np.linspace(0, len(vr)-1, T, dtype=int)
        try:                                        # ← 第二層保護
            frames = vr.get_batch(idx).asnumpy()
        except Exception:
            frames = np.stack([vr[i].asnumpy() for i in idx])
        times = ",".join([f"{i/vr.get_avg_fps():.2f}s" for i in idx])
        return frames, times, len(vr)/vr.get_avg_fps()
    except Exception as e:
        print("[WARN] broken or unreadable:", path, "|", e)
        return None, None, None

# -----------------------------------------------------------------------------
# 2. Dataset class (support both dict‑map JSON & list‑of‑objects JSON)
# -----------------------------------------------------------------------------

class SignLanguageVideoDataset(Dataset):
    """Dataset yielding video frames & prompt‑specific meta for sign‑language translation.

    JSON formats supported:
    ①  {"file1.mp4": "translation", "file2.mp4": "..."}
    ②  [ {"video_serial_number": "xxx", "input": "...", "output": "..."}, ... ]
        • video file assumed to be <video_serial_number>.mp4 (configurable)
        • label = input + " " + output
    """

    def __init__(
        self,
        video_dir: str,
        label_json: str,
        tokenizer,
        image_processor,
        *,
        T: int = 16,
        tpl: str = "qwen_1_5",
        video_ext: str = ".mp4",
    ):
        super().__init__()

        with open(label_json, "r", encoding="utf-8") as f:
            raw_obj: Union[Dict[str, str], List[Dict[str, Any]]] = json.load(f)

        # 2‑A  Key‑value dict (old format)
        if isinstance(raw_obj, dict):
            self.label_map: Dict[str, str] = raw_obj
        # 2‑B  List of objects (new format)
        elif isinstance(raw_obj, list):
            tmp = {}
            for item in raw_obj:
                vsn = item["video_serial_number"]
                fn = f"{vsn}{video_ext}"
                # combine input + output (instruction 已在 prompt)
                tmp[fn] = item.get("input", "") + " " + item.get("output", "")
            self.label_map = tmp
        else:
            raise ValueError("Unsupported JSON structure for labels.")

        self.video_paths: List[str] = [os.path.join(video_dir, fn) for fn in self.label_map]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.T = T
        self.tpl = tpl

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vpath = self.video_paths[idx]
        video_np, frame_times, video_len = load_video(vpath, self.T)
        label = self.label_map[os.path.basename(vpath)]
        return {
            "video_np": video_np,
            "frame_times": frame_times,
            "video_len": video_len,
            "label": label,
            "video_path": vpath,
        }

# -----------------------------------------------------------------------------
# 3. Collate‑function factory (unchanged logic)
# -----------------------------------------------------------------------------

def make_collate_fn(tokenizer, image_processor, *, tpl: str = "qwen_1_5", device: str = "cpu"):
    pad_id = tokenizer.pad_token_id

    def collate_fn(samples: List[Dict[str, Any]]):
        batch_video_tensors: List[torch.Tensor] = []
        batch_input_ids:   List[torch.Tensor] = []   # padded later
        batch_prompt_ids:  List[torch.Tensor] = []   # raw (no pad)  ← 可選
        batch_prompts:     List[str] = []            # prompt text   ← 用來列印
        batch_labels:      List[str] = []
        max_len = 0
        '''
        print("\n==== 🔍 samples debug ====")
        print(f"type(samples): {type(samples)}")
        print(f"len(samples): {len(samples)}")
        for i, s in enumerate(samples):
            print(f"\n--- Sample {i} ---")
            for k, v in s.items():
                if isinstance(v, np.ndarray):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"{k}: {v if isinstance(v, str) else type(v)}")'''
        for s in samples:
            # (a) video -> tensor (float32, CPU)
            
            pil_frames = [Image.fromarray(f).resize((224, 224), Image.BICUBIC)
                          for f in s["video_np"]]
            video_pt = image_processor.preprocess(
                pil_frames, return_tensors="pt"
            )["pixel_values"].to(dtype=torch.float32)   # 留在 CPU
            batch_video_tensors.append(video_pt)

            # (b) build prompt
            instr = (
                f"這段影片長度為 {s['video_len']:.2f} 秒，"
                f"我們從中平均取樣 {len(video_pt)} 幀（時間點：{s['frame_times']}）。"
                "影片中有人正在以手語（Sign Language）表達完整句子。"
                "請你扮演專業的手語翻譯員，"
                "仔細觀看這些影格並只關注手語內容，"
                "將其翻譯成流暢且意義完整的【繁體中文書面語】。"
            )
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + instr
            batch_prompts.append(prompt)

            # (c) tokenizer -> ids
            conv = copy.deepcopy(conv_templates[tpl])
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(
                conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            ).squeeze(0)           # shape (L,)
            batch_prompt_ids.append(ids)
            max_len = max(max_len, ids.size(0))

            batch_labels.append(s["label"])

        # (d) pad input_ids to (B, max_len)
        B = len(samples)
        input_ids = torch.full((B, max_len), pad_id,
                               dtype=torch.long, device=device)
        for i, ids in enumerate(batch_prompt_ids):
            input_ids[i, :ids.size(0)] = ids
        batch_input_ids = input_ids  # rename for clarity

        return {
            "input_ids":   batch_input_ids,   # Tensor  (B, L)
            "images":      batch_video_tensors,  # List[Tensor] (T,3,224,224)
            "labels":      batch_labels,      # List[str]
            "prompts":     batch_prompts,     # List[str]  ← 新增
            "prompt_ids":  batch_prompt_ids,  # List[Tensor] (可選)
        }

    return collate_fn


# -----------------------------------------------------------------------------
# 4. DataLoader helper (unchanged API)
# -----------------------------------------------------------------------------

def build_dataloader(
    video_dir: str,
    label_json: str,
    tokenizer,
    image_processor,
    *,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    T: int = 16,
    device: str = "cpu",
) -> DataLoader:
    dataset = SignLanguageVideoDataset(
        video_dir, label_json, tokenizer, image_processor, T=T
    )
    collate_fn = make_collate_fn(tokenizer, image_processor, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)

# -----------------------------------------------------------------------------
# If run standalone, quick smoke test (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        "lmms-lab/LLaVA-Video-7B-Qwen2",
        None,
        "llava_qwen",
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="sdpa"
    )
    
    # 先建立 DataLoader：worker 全在 CPU，避免 CUDA fork 問題
    dl = build_dataloader(
        video_dir="data/chinese/train",          # 影片資料夾
        label_json="data/chinese/train.json",    # 標註 JSON
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=1,
        device="cpu",        # collate_fn 端只用 CPU
        num_workers=4
    )

    for b in dl:
        print("input_ids shape:", b["input_ids"].shape)
        print("input_ids shape:", b["input_ids"])
        print("video shape     :", b["images"][0].shape)
        print("prompt text     :", b["prompts"][0][:120], "...")
        print("label           :", b["labels"][0], "...")
        break