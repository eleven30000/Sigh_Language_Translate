from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
import jieba
# 設定輸入路徑
input_json = "./data/score/BLUE.json"

# 載入 JSON
with open(input_json, encoding="utf-8") as f:
    data = json.load(f)

# 擷取預測與參考（中文斷詞）
hypotheses = [' '.join(jieba.cut(item["prediction"])) for item in data]
references = [[' '.join(jieba.cut(item["reference"]))] for item in data]

# nltk 格式：List[List[str]], List[List[List[str]]]
hyps_tokenized = [h.split() for h in hypotheses]
refs_tokenized = [[r.split() for r in ref_list] for ref_list in references]

print(f"預測數量：{len(hyps_tokenized)}")
print(f"參考數量：{len(refs_tokenized)}")
print(f"hyps_tokenized：{hyps_tokenized}")
print(f"refs_tokenized：{refs_tokenized}")


smooth = SmoothingFunction().method1

# 計算 BLEU-N
bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1, 0, 0, 0), smoothing_function=smooth)
bleu2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
bleu3 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

print(f"BLEU-1: {bleu1 * 100:.2f}")
print(f"BLEU-2: {bleu2 * 100:.2f}")
print(f"BLEU-3: {bleu3 * 100:.2f}")
print(f"BLEU-4: {bleu4 * 100:.2f}")
