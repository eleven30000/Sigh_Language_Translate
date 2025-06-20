#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run inference on test.mp4 using LLaVA-Video-7B-Qwen2 + SimpleAdapter.
執行：
    python inference.py
"""

import os, copy, warnings, torch, numpy as np
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from data_process import load_video
from adapter_init import SimpleAdapter          # ← 直接重用訓練時的 Adapter

warnings.filterwarnings("ignore")
os.environ["ATTN_IMPLEMENTATION"] = "sdpa"

# -------- 路徑 / 參數 --------
VIDEO_PATH   = "data/test.mp4"
ADAPTER_PATH = "checkpoints/adapter_init.pt"
MODEL_NAME   = "lmms-lab/LLaVA-Video-7B-Qwen2"
DEVICE       = "cuda"
FRAMES       = 16
# ----------------------------


def attach_adapter_to_model(base_model, adapter_state):
    hidden_size = base_model.config.hidden_size
    model_dtype  = next(base_model.parameters()).dtype
    model_device = next(base_model.parameters()).device

    adapter = SimpleAdapter(hidden_size).to(device=model_device,
                                            dtype=model_dtype)
    adapter.load_state_dict(adapter_state)
    adapter.eval()

    orig_forward = base_model.forward

    def patched_forward(input_ids=None,
                        attention_mask=None,      # 必須顯式列出！
                        images=None,
                        modalities=None,
                        **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        out = orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            modalities=modalities,
            **kwargs
        )
        hidden = out.hidden_states[-1]
        hidden = adapter(hidden)
        out.logits = base_model.lm_head(hidden)
        return out

    base_model.forward = patched_forward
    base_model.adapter = adapter
    return base_model




def build_prompt(video_len: float, frame_times: str, T: int) -> str:
    instr = (
        f"這段影片長度為 {video_len:.2f} 秒，"
        f"我們從中平均取樣 {T} 幀（時間點：{frame_times}）。"
        "影片中有人正在以手語（Sign Language）表達完整句子。"
        "請你扮演專業的手語翻譯員，"
        "仔細觀看這些影格並只關注手語內容，"
        "將其翻譯成流暢且意義完整的【繁體中文書面語】。"
    )
    return DEFAULT_IMAGE_TOKEN + "\n" + instr


def main():
    # 1. 讀模型
    tokenizer, base_model, image_processor, _ = load_pretrained_model(
        MODEL_NAME, None, "llava_qwen",
        torch_dtype="bfloat16", device_map="auto",
        attn_implementation="sdpa"
    )
    base_model.eval()

    # 2. 接上 adapter
    adapter_state = torch.load(ADAPTER_PATH, map_location="cpu")
    base_model = attach_adapter_to_model(base_model, adapter_state)
    print("✅ Adapter loaded & attached.")

    # 3. 讀影片
    video_np, frame_times, video_len = load_video(VIDEO_PATH, FRAMES)
    pil_frames = [Image.fromarray(f).resize((224,224), Image.BICUBIC)
                  for f in video_np]

    video_pt = image_processor.preprocess(
        pil_frames, return_tensors="pt"
    )["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
    video_pt = [video_pt]

    # 4. prompt & input_ids
    prompt = build_prompt(video_len, frame_times, video_pt[0].size(0))
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    # 5. 生成
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_ids = base_model.generate(
            input_ids,
            images         = video_pt,
            modalities     = ["video"],
            do_sample      = False,
            temperature    = 0.0,
            max_new_tokens = 256
        )

    translation = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    print("\n=== 翻譯結果 ===\n", translation)


if __name__ == "__main__":
    main()
