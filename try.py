import os, copy, warnings, torch, numpy as np
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
device = "cuda"
torch.cuda.empty_cache()

# 1️⃣ 載模型（保持 bfloat16，這版最穩定也省記憶體）
tokenizer, model, image_processor, _ = load_pretrained_model(
    "lmms-lab/LLaVA-Video-7B-Qwen2",
    None,
    "llava_qwen",
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation="sdpa"
)
model.eval()

# 2️⃣ 讀影片（最多 16 幀，避免撐爆顯存；可自行調小）
def load_video(path, T=16):
    vr = VideoReader(path, ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, T, dtype=int)
    times = ",".join([f"{i/vr.get_avg_fps():.2f}s" for i in idx])
    return vr.get_batch(idx).asnumpy(), times, len(vr)/vr.get_avg_fps()

video_np, frame_times, video_len = load_video("data/test.mp4", 16)

pil_frames = [Image.fromarray(f).resize((224, 224), Image.BICUBIC)
              for f in video_np]

# 3️⃣ 影像前處理 → **直接轉成 bfloat16**（重點修正）
video_pt = image_processor.preprocess(
    pil_frames,
    return_tensors="pt"           # 只剩這個參數合法
)["pixel_values"].to(device, dtype=torch.bfloat16)

video_pt = [video_pt]          # llava 的 API 期望 list

# 4️⃣ 組 prompt
tpl    = "qwen_1_5"
instr = (
    f"這段影片長度為 {video_len:.2f} 秒，"
    f"我們從中平均取樣 {len(video_pt[0])} 幀（時間點：{frame_times}）。"
    "影片中有人正在以手語（Sign Language）表達完整句子。"
    "請你扮演專業的手語翻譯員，"
    "仔細觀看這些影格並只關注手語內容，"
    "將其翻譯成流暢且意義完整的【繁體中文書面語】。"
)
prompt = DEFAULT_IMAGE_TOKEN + "\n" + instr
conv   = copy.deepcopy(conv_templates[tpl])
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(
               conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
               return_tensors="pt").unsqueeze(0).to(device)

# 5️⃣ 生成（關梯度 + AMP）；把輸出長度縮到 512 免得又炸
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out_ids = model.generate(
        input_ids,
        images      = video_pt,
        modalities  = ["video"],
        do_sample   = False,
        temperature = 0.0,
        max_new_tokens = 256
    )

print(tokenizer.decode(out_ids[0], skip_special_tokens=True).strip())

del model, video_pt, input_ids
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
# ─────────────────────────────────────────────────────────────