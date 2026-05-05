import h5py
import json
import torch
from evaluate import get_exo_cameras, adjust_prompt, process_frame, register_llava_model

register_llava_model()
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.conversation import default_conversation
from llava.constants import IMAGE_TOKEN_INDEX
from transformers import CLIPImageProcessor

model_path = "data/model/model"
model_name = get_model_name_from_path(model_path)
tokenizer, model, _, ctx = load_pretrained_model(
    model_path, "liuhaotian/llava-v1.5-7b", model_name,
    load_8bit=False, load_4bit=False, device_map="auto",
)
model.config.tokenizer_padding_side = "left"

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device="cuda", dtype=torch.float16)

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
device = next(model.parameters()).device

with open("data/test_1perm_Falsetemp_Falsetempaug_EgoExOR_5k_samples_drophistory0.5.json") as f:
    samples = json.load(f)

sample = None
for s in samples:
    if s["hdf5_indices"]["surgery_type"] == "MISS":
        sample = s
        break

h = sample["hdf5_indices"]
take = f"data/{h['surgery_type']}/{h['procedure_id']}/take/{h['take_id']}"
exo_ids, exo_names = get_exo_cameras("data/hdf5/data/egoexor_miss.h5", take)
print(f"Take: {take}, exo cameras: {exo_names} ({exo_ids})")
print(f"Frame idx: {h['frame_idx']}")

with h5py.File("data/hdf5/data/egoexor_miss.h5", "r") as f:
    frame_rgb = f[f"{take}/frames/rgb"][h["frame_idx"]]

print(f"frame_rgb shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}, range: [{frame_rgb.min()}, {frame_rgb.max()}]")

imgs = []
for sid in exo_ids:
    t = torch.from_numpy(frame_rgb[sid]).float()
    print(f"  Camera {sid}: shape={t.shape}, range=[{t.min():.1f}, {t.max():.1f}], all_zero={t.max()==0}")
    proc = process_frame(t, image_processor)
    if proc is not None:
        imgs.append(proc)
print(f"Processed {len(imgs)} images, each shape: {imgs[0].shape}")

prompt = adjust_prompt(sample["conversations"][0]["value"], len(imgs))
print(f"Prompt: {prompt[:300]}")

conv = default_conversation.copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
full_prompt = conv.get_prompt()

input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
print(f"input_ids shape: {input_ids.shape}")
num_img_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
print(f"Num image tokens: {num_img_tokens}, Num images: {len(imgs)}")

flat_images = [img.unsqueeze(0).to(device, dtype=torch.float16) for img in imgs]

with torch.inference_mode():
    output_ids = model.generate(
        inputs=input_ids,
        images=flat_images,
        do_sample=False,
        use_cache=True,
        max_new_tokens=300,
    )

output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
print(f"Raw output: [{output}]")
print(f"GT: [{sample['conversations'][1]['value'][:200]}]")
