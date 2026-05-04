#!/usr/bin/env python3
"""
Evaluate EgoExOR model on MISS test split using exocentric RGB frames only.

Usage:
    python evaluate.py \
        --model_path data/model/model \
        --test_json data/test_1perm_Falsetemp_Falsetempaug_EgoExOR_5k_samples_drophistory0.5.json \
        --hdf5_path data/egoexor_miss.h5 \
        --output_csv eval_miss_exo_results.csv
"""

import argparse
import csv
import json
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

EXO_SOURCES = {
    "or_light",
    "microscope",
    "external_1",
    "external_2",
    "external_3",
    "external_4",
    "external_5",
    "simstation",
}

scene_graph_name_to_vocab_idx = {
    "anesthetist": 0, "anesthesia_equipment": 1, "antiseptic": 2, "assistant": 3,
    "bin": 4, "body_marker": 5, "circulator": 6, "cotton": 7, "curette": 8,
    "dressing_material": 9, "forceps": 10, "gloves": 11, "head_surgeon": 12,
    "health_monitor": 13, "herbal_disk": 14, "instrument_table": 15, "instruments": 16,
    "microscope": 17, "microscope_controller": 18, "microscope_eye": 19,
    "microscope_screen": 20, "needle": 21, "operating_room": 22, "operating_table": 23,
    "patient": 24, "scalpel": 25, "scissors": 26, "syringe": 27, "tissue_mark": 28,
    "tissue_paper": 29, "ultrasound_gel": 30, "ultrasound_machine": 31,
    "ultrasound_probe": 32, "ultrasound_screen": 33, "unsterile_instruments": 34,
    "vertebrae": 35,
    "anaesthetising": 36, "applying": 37, "aspirating": 38, "looking": 39,
    "closeto": 40, "controlling": 41, "cutting": 42, "disinfection": 43, "dressing": 44,
    "dropping": 45, "entering": 46, "holding": 47, "injecting": 48, "inserting": 49,
    "lyingon": 50, "manipulating": 51, "positioning": 52, "preparing": 53, "removing": 54,
    "scanning": 55, "touching": 56, "wearing": 57,
}

entity_synonyms = {
    "operating_table": ["operation_table", "operating_table"],
    "anesthetist": ["anaesthetist"],
    "microscope": ["microcope"],
}
relation_synonyms = {
    "closeto": ["closeTo"],
    "lyingon": ["lyingOn"],
    "looking": ["looking", "checking"],
    "anaesthetising": ["anesthasing"],
}


def _reverse_synonyms(syn):
    out = {}
    for k, vs in syn.items():
        for v in vs:
            out[v] = k
    return out


_rev_ent = _reverse_synonyms(entity_synonyms)
_rev_rel = _reverse_synonyms(relation_synonyms)


def _name_to_idx(name):
    name = name.lower()
    name = _rev_rel.get(name, name)
    name = _rev_ent.get(name, name)
    return scene_graph_name_to_vocab_idx[name]


def parse_triplets(text):
    triplets = []
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    if "<SG>" in text and "</SG>" in text and text.index("<SG>") < text.index("</SG>"):
        parts = text.split("<SG>")[1].split("</SG>")[0].strip().split(";")
    else:
        parts = text.split(";")
    for t in parts:
        t = t.replace(".", "").replace("</s>", "").replace("<s>", "").strip()
        if not t:
            continue
        elems = [e.strip() for e in t.split(",")]
        if len(elems) != 3:
            continue
        s, o, p = elems
        s = _rev_ent.get(s, s)
        o = _rev_ent.get(o, o)
        p = _rev_rel.get(p, p)
        triplets.append((s, p, o))
    return triplets


def map_triplets(triplets):
    out = []
    for s, p, o in triplets:
        try:
            out.append((_name_to_idx(s.replace(" ", "_")), _name_to_idx(p), _name_to_idx(o.replace(" ", "_"))))
        except Exception:
            pass
    return out


def get_exo_cameras(h5_path, take_path):
    ids, names = [], []
    with h5py.File(h5_path, "r") as f:
        grp = f.get(f"{take_path}/sources")
        if grp is None:
            return ids, names
        for i in range(grp.attrs.get("source_count", 0)):
            name = grp.attrs.get(f"source_{i}", "")
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if name in EXO_SOURCES:
                ids.append(i)
                names.append(name)
    return ids, names


def adjust_prompt(text, n_images):
    current = text.count("<image>")
    if current == n_images:
        return text
    prompt = re.sub(r"(<image>\s*)+", "", text).strip()
    return " ".join(["<image>"] * n_images) + " " + prompt


def expand2square(img, bg):
    w, h = img.size
    if w == h:
        return img
    dim = max(w, h)
    r = Image.new(img.mode, (dim, dim), bg)
    r.paste(img, ((dim - w) // 2, (dim - h) // 2))
    return r


def process_frame(frame_tensor, image_processor):
    """Process a single [H,W,3] uint8 tensor into [C,H,W] for LLaVA."""
    arr = frame_tensor.cpu().numpy()
    if arr.max() == 0:
        return None
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")
    bg = tuple(int(x * 255) for x in image_processor.image_mean)
    pil = expand2square(pil, bg)
    out = image_processor.preprocess(pil, return_tensors="pt")["pixel_values"]
    return out.squeeze(0)


def register_llava_model():
    """Register LlavaLlamaForCausalLM for 'llava' model_type (fixes transformers 4.37+ conflict)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

    LlavaConfig.model_type = "llava"
    AutoConfig.register("llava", LlavaConfig, exist_ok=True)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, exist_ok=True)


def load_model(model_path):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    register_llava_model()

    print(f"Loading model from {model_path}...")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        "liuhaotian/llava-v1.5-7b",
        model_name,
        load_8bit=False,
        load_4bit=True,
        device_map="auto",
    )
    model.config.tokenizer_padding_side = "left"
    return tokenizer, model, image_processor


def load_miss_samples(test_json_path):
    with open(test_json_path) as f:
        all_samples = json.load(f)
    samples = [deepcopy(s) for s in all_samples if s["hdf5_indices"]["surgery_type"] == "MISS"]
    print(f"Filtered {len(samples)} MISS test samples from {len(all_samples)} total")
    return samples


def run_inference(tokenizer, model, image_processor, samples, hdf5_path, batch_size=1):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import default_conversation
    from llava.mm_utils import tokenizer_image_token

    device = next(model.parameters()).device
    csv_rows = []
    exo_cache = {}

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch_samples = samples[i : i + batch_size]
        bs = len(batch_samples)
        all_images = []
        all_prompts = []
        all_gt = []

        with h5py.File(hdf5_path, "r") as f:
            for s in batch_samples:
                h = s["hdf5_indices"]
                take = f"data/{h['surgery_type']}/{h['procedure_id']}/take/{h['take_id']}"
                fidx = h["frame_idx"]

                if take not in exo_cache:
                    exo_cache[take] = get_exo_cameras(hdf5_path, take)
                exo_ids, _ = exo_cache[take]

                frame_rgb = f[f"{take}/frames/rgb"][fidx]
                imgs = []
                for sid in exo_ids:
                    proc = process_frame(torch.from_numpy(frame_rgb[sid]).float(), image_processor)
                    if proc is not None:
                        imgs.append(proc)

                if not imgs:
                    imgs = [torch.zeros(3, 336, 336)]

                all_images.append(torch.stack(imgs))

                prompt = adjust_prompt(s["conversations"][0]["value"], len(imgs))
                conv = deepcopy(default_conversation)
                conv.append_message("human", prompt)
                conv.append_message("gpt", None)
                all_prompts.append(conv.get_prompt())
                all_gt.append(s["conversations"][1]["value"])

        if bs == 1:
            input_ids = tokenizer_image_token(
                all_prompts[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
        else:
            ids_list = [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in all_prompts]
            inv = [torch.flip(ids, [0]) for ids in ids_list]
            pad = torch.nn.utils.rnn.pad_sequence(inv, batch_first=True, padding_value=tokenizer.pad_token_id)
            input_ids = torch.flip(pad, [1]).to(device)

        images_batch = [imgs.to(device, dtype=torch.float16) for imgs in all_images]

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=images_batch,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
            )

        if bs == 1:
            texts = [tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()]
        else:
            texts = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

        for j, (output, gt_text) in enumerate(zip(texts, all_gt)):
            s = batch_samples[j]
            h = s["hdf5_indices"]

            pred_t = parse_triplets(output)
            gt_t = parse_triplets(gt_text)
            pred_m = set(map_triplets(pred_t))
            gt_m = set(map_triplets(gt_t))

            if not gt_m and not pred_m:
                f1 = 1.0
            elif not gt_m:
                f1 = 0.0
            else:
                tp = len(gt_m & pred_m)
                prec = tp / len(pred_m) if pred_m else 0.0
                rec = tp / len(gt_m)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            csv_rows.append({
                "sample_id": f"{h['surgery_type']}_{h['procedure_id']}_{h['take_id']}_{h['frame_idx']}",
                "procedure": h["surgery_type"],
                "phase": h["procedure_id"],
                "take": h["take_id"],
                "frame_idx": h["frame_idx"],
                "ground_truth": "; ".join(f"{s},{o},{p}" for s, p, o in gt_t),
                "prediction": "; ".join(f"{s},{o},{p}" for s, p, o in pred_t),
                "gt_count": len(gt_m),
                "pred_count": len(pred_m),
                "true_positives": len(gt_m & pred_m),
                "frame_f1": round(f1, 4),
                "raw_output": output[:1000],
            })

    return csv_rows


def save_results(csv_rows, output_csv):
    output_csv = Path(output_csv)
    if not csv_rows:
        print("No results to save.")
        return

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    tp = sum(r["true_positives"] for r in csv_rows)
    total_gt = sum(r["gt_count"] for r in csv_rows)
    total_pred = sum(r["pred_count"] for r in csv_rows)
    prec = tp / total_pred if total_pred else 0.0
    rec = tp / total_gt if total_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    avg_f1 = np.mean([r["frame_f1"] for r in csv_rows])

    print(f"\n{'=' * 60}")
    print("  MISS Exocentric RGB-Only Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Total frames:          {len(csv_rows)}")
    print(f"  Precision:             {prec:.4f}")
    print(f"  Recall:                {rec:.4f}")
    print(f"  Overall F1 (triplet):  {f1:.4f}")
    print(f"  Average Frame F1:      {avg_f1:.4f}")
    print(f"{'=' * 60}")
    print(f"  CSV:  {output_csv}")

    summary = output_csv.with_suffix(".summary.txt")
    with open(summary, "w") as f:
        f.write(f"MISS Exocentric RGB-Only Evaluation\n{'=' * 40}\n")
        f.write(f"Frames: {len(csv_rows)}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\nAvg Frame F1: {avg_f1:.4f}\n")
    print(f"  Summary: {summary}")


def main():
    p = argparse.ArgumentParser(description="Evaluate EgoExOR on MISS test split (exocentric RGB only)")
    p.add_argument("--model_path", required=True, help="Path to model checkpoint dir")
    p.add_argument("--test_json", required=True, help="Path to test samples JSON")
    p.add_argument("--hdf5_path", required=True, help="Path to merged MISS HDF5 file")
    p.add_argument("--output_csv", default="eval_miss_exo_results.csv")
    p.add_argument("--batch_size", type=int, default=1)
    args = p.parse_args()

    tokenizer, model, image_processor = load_model(args.model_path)
    samples = load_miss_samples(args.test_json)
    rows = run_inference(tokenizer, model, image_processor, samples, args.hdf5_path, args.batch_size)
    save_results(rows, args.output_csv)


if __name__ == "__main__":
    main()
