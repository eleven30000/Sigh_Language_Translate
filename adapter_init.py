#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune LLaVA-Video-7B-Qwen2 with a lightweight adapter on sign-language data.
執行：
    python adapter_init.py
"""

import os
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_process import build_dataloader
from llava.model.builder import load_pretrained_model

# ==================== 參數設定 ====================

VIDEO_DIR   = "data/chinese/train"          # 影片資料夾
LABEL_JSON  = "data/chinese/train.json"     # 標註 JSON
OUTPUT_DIR  = "checkpoints"                # 儲存目錄
MODEL_NAME  = "lmms-lab/LLaVA-Video-7B-Qwen2"

BATCH_SIZE  = 1
NUM_WORKERS = 4
EPOCHS      = 50
LR          = 5e-4
FRAMES      = 16
DEVICE      = "cuda"                        # or "cpu"

# =================================================


# ----------------------------- Adapter Layer -----------------------------
class SimpleAdapter(nn.Module):
    """Bottleneck adapter: down → ReLU → up；殘差加回輸入"""

    def __init__(self, hidden_size: int, bottleneck: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)       # 近似 identity

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


# --------------------------- Wrapper Model ------------------------------
class LLaVAWithAdapter(nn.Module):
    """凍結 backbone，只訓練 adapter (+ lm_head)。"""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad_(False)

        hidden_size = self.base.config.hidden_size
        self.adapter = SimpleAdapter(hidden_size).to(torch.bfloat16) 
        self.lm_head = base_model.lm_head      # 共用權重（可微分）

    def forward(self, input_ids, images, modalities, labels=None):
        outputs = self.base(
            input_ids=input_ids,
            images=images,
            modalities=modalities,
            output_hidden_states=True,   # ★ 改成 False
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1][:, :input_ids.size(1), :]  # 對齊
        hidden = self.adapter(hidden)
        logits = self.lm_head(hidden)                  # (B, L, vocab)
        #print("hidden.shape:", hidden.shape)
        #print("logits.shape:", logits.shape)
        loss = None

        if labels is not None:
            '''
            # Step 1: 自動補長 labels 成跟 logits 對齊
            B, L_logits = logits.shape[:2]     # e.g. (1, 3823)
            B, L_labels = labels.shape         # e.g. (1, 463)

            if L_logits != L_labels:
                new_labels = torch.full(
                    (B, L_logits), -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                # 將原始 labels 貼到尾端，讓答案對齊序列末尾
                new_labels[:, -L_labels:] = labels
                labels = new_labels'''
        
            #print("labels:", labels)
            #print("labels shape:", labels.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            #print(shift_logits.shape)
            #print(shift_labels.shape)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return {"loss": loss, "logits": logits}


# -------------------- Helper: build labels tensor -----------------------
def build_prompt_and_labels(prompt_ids: torch.Tensor,
                            answer: str,
                            tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """將 prompt_ids 與答案 token 接起來，生成 labels（prompt 部分 -100）。"""
    device = prompt_ids.device
    pad_id = tokenizer.pad_token_id

    valid_len = (prompt_ids != pad_id).sum().item()
    prompt_ids = prompt_ids[:valid_len]
    #print("prompt_ids shape:", prompt_ids.shape)
    answer_ids = tokenizer(
        answer, add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.squeeze(0).to(device)
    #print("prompt_ids:", prompt_ids)
    #print("prompt_ids shape:", prompt_ids.shape)
    #print("answer_ids :", answer_ids)
    #print("answer_ids shape:", answer_ids.shape)

    MAX_ANS_TOK = 256
    if answer_ids.size(0) > MAX_ANS_TOK:
        answer_ids = answer_ids[-MAX_ANS_TOK:]

    eos = torch.tensor([tokenizer.eos_token_id], device=device)
    answer_ids = torch.cat([answer_ids, eos], dim=0)
    #print("answer_ids 2:", answer_ids)
    #print("answer_ids shape2:", answer_ids.shape)
    combined = torch.cat([prompt_ids, answer_ids], dim=0)
    labels = torch.full_like(combined, -100)
    labels[prompt_ids.size(0):] = combined[prompt_ids.size(0):]
    #print("labels:", labels)
    #print("labels shape:", labels.shape)
    #print("combined:", combined)
    #print("combined shape:", combined.shape)
    #labels是combined的label，combined是prompt_ids和answer_ids的結合
    return combined, labels


# ------------------------------- Train ----------------------------------
def main():
    # 1. 下載 / 載入模型
    tokenizer, base_model, image_processor, _ = load_pretrained_model(
        "lmms-lab/LLaVA-Video-7B-Qwen2",
        None,
        "llava_qwen",
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="sdpa"
    )
    base_model.eval()
    model = LLaVAWithAdapter(base_model).to(DEVICE)

    # 2. DataLoader（worker 僅 CPU）
    dl = build_dataloader(
        VIDEO_DIR, 
        LABEL_JSON,
        tokenizer, 
        image_processor,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device="cpu",          # collate_fn 留在 CPU
        T=FRAMES,
    )

    # 3. Optimiser / Scheduler
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optim, T_max=EPOCHS * len(dl))

    # 4. 訓練迴圈
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            # -- 構造 input_ids & label_ids --
            prompt_ids_list, label_ids_list, max_len = [], [], 0
            for i in range(len(batch["labels"])):
                combined, labels = build_prompt_and_labels(
                    batch["input_ids"][i], batch["labels"][i], tokenizer
                )
                prompt_ids_list.append(combined)
                label_ids_list.append(labels)
                max_len = max(max_len, combined.size(0))

            pad_id = tokenizer.pad_token_id
            B = len(prompt_ids_list)
            input_ids = torch.full((B, max_len), pad_id,
                                   dtype=torch.long, device=DEVICE)
            label_ids = torch.full_like(input_ids, -100)
            for i, (ids, lbl) in enumerate(zip(prompt_ids_list, label_ids_list)):
                input_ids[i, :ids.size(0)] = ids.to(DEVICE)
                label_ids[i, :lbl.size(0)] = lbl.to(DEVICE)
            #print("!!input_ids:", input_ids)
            #print("!!input_ids shape:", input_ids.shape)
            #print("!!label_ids:", label_ids)
            #print("!!label_ids shape:", label_ids.shape)
            # -- 處理影像 --
            images = [v.to(DEVICE, dtype=torch.bfloat16) for v in batch["images"]]

            out = model(input_ids=input_ids,
                        images=images,
                        modalities=["video"],
                        labels=label_ids)
            #print("out:", out)
            loss = out["loss"]
            #print("loss:", loss)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()
            pbar.set_postfix({"loss": loss.item()})

    # 5. 儲存 adapter
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.adapter.state_dict(), os.path.join(OUTPUT_DIR, "adapter_init.pt"))
    print("✅ Training done. Adapter saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
