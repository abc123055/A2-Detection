"""
VLM (Vision Language Model) module for generating frame-level scene descriptions.

Supports multiple backends:
  - qwen35:  Qwen3.5-35B-A3B (~35GB VRAM, 3B active MoE), latest and best
  - qwen2vl: Qwen2.5-VL-32B (~35GB VRAM fp16), strong general VLM
  - cogvlm:  CogVLM (~15GB VRAM bfloat16), good for actions
  - blip:    BLIP-base (~1GB VRAM fp16), lightweight

Usage:
    python vlm.py --data Mydataset
    python vlm.py --data Mydataset --backend qwen2vl
"""

import os
import torch
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_free_vram_gb():
    """Get free VRAM in GB. Returns 0 if no GPU available."""
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


# ──────────────────────── Model Loading ────────────────────────

def _load_blip():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_path = os.path.join(os.path.dirname(__file__), "blip-image-captioning-base")
    print(f"Loading BLIP model from {blip_path}...")
    processor = BlipProcessor.from_pretrained(blip_path)
    model = BlipForConditionalGeneration.from_pretrained(
        blip_path, torch_dtype=torch.float16
    ).to(device).eval()
    return model, processor


def _load_cogvlm():
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    cogvlm_path = os.path.join(os.path.dirname(__file__), "cogvlm-chat")
    vicuna_path = os.path.join(os.path.dirname(__file__), "vicuna-7b-v1.5")
    print(f"Loading CogVLM model from {cogvlm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(vicuna_path)
    model = AutoModelForCausalLM.from_pretrained(
        cogvlm_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    return model, tokenizer


def _load_qwen2vl():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    qwen2vl_path = os.path.join(os.path.dirname(__file__), "Qwen2.5-VL-32B-Instruct")
    print(f"Loading Qwen2.5-VL model from {qwen2vl_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen2vl_path, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(qwen2vl_path)
    return model, processor


def _load_qwen35():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qwen35_path = os.path.join(os.path.dirname(__file__), "Qwen3.5-35B-A3B")
    print(f"Loading Qwen3.5-35B-A3B model from {qwen35_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        qwen35_path, torch_dtype="auto", device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(qwen35_path)
    return model, processor


def _unload(model, processor_or_tokenizer):
    del model, processor_or_tokenizer
    torch.cuda.empty_cache()


# ──────────────────────── Inference ────────────────────────

def _describe_blip(frame_paths, model, processor, prompt=None, batch_size=4):
    prompt = prompt or "a photo of"
    descriptions = []
    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        if prompt:
            inputs = processor(images=images, text=[prompt] * len(images),
                               return_tensors="pt", padding=True)
        else:
            inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device, torch.float16)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80)
        captions = processor.batch_decode(outputs, skip_special_tokens=True)
        descriptions.extend(captions)

        done = min(i + batch_size, len(frame_paths))
        if (i // batch_size) % 10 == 0 or done == len(frame_paths):
            print(f"  [{done}/{len(frame_paths)}] {captions[-1][:80]}...")

    return descriptions


def _describe_cogvlm(frame_paths, model, tokenizer, prompt=None):
    prompt = prompt or "Describe this image in detail."
    descriptions = []
    for idx, path in enumerate(frame_paths):
        image = Image.open(path).convert("RGB")
        inputs = model.build_conversation_input_ids(
            tokenizer, query=prompt, images=[image]
        )
        input_ids = inputs['input_ids'].unsqueeze(0).to(device)
        token_type_ids = inputs['token_type_ids'].unsqueeze(0).to(device)
        attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
        images_tensor = [[inputs['images'][0].to(device).to(torch.bfloat16)]]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, token_type_ids=token_type_ids,
                attention_mask=attention_mask, images=images_tensor,
                max_new_tokens=512, do_sample=False,
            )
        resp = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        descriptions.append(resp.strip())

        if idx % 10 == 0 or idx == len(frame_paths) - 1:
            print(f"  [{idx + 1}/{len(frame_paths)}] {resp[:80]}...")

    return descriptions


def _describe_qwen2vl(frame_paths, model, processor, prompt=None):
    from qwen_vl_utils import process_vision_info
    prompt = prompt or "Describe this image in detail, including human activities and objects."
    descriptions = []
    for idx, path in enumerate(frame_paths):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(path)}"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        desc = processor.decode(generated_ids, skip_special_tokens=True).strip()
        descriptions.append(desc)

        if idx % 10 == 0 or idx == len(frame_paths) - 1:
            print(f"  [{idx + 1}/{len(frame_paths)}] {desc[:80]}...")

    return descriptions


def _describe_qwen35(frame_paths, model, processor, prompt=None):
    prompt = prompt or "Describe this image in detail, including human activities and objects."
    descriptions = []
    for idx, path in enumerate(frame_paths):
        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        desc = processor.decode(generated_ids, skip_special_tokens=True).strip()
        descriptions.append(desc)

        if idx % 10 == 0 or idx == len(frame_paths) - 1:
            print(f"  [{idx + 1}/{len(frame_paths)}] {desc[:80]}...")

    return descriptions


# ──────────────────────── Public API ────────────────────────

BACKENDS = {
    'blip':    {'load': _load_blip,    'describe': _describe_blip},
    'cogvlm':  {'load': _load_cogvlm,  'describe': _describe_cogvlm},
    'qwen2vl': {'load': _load_qwen2vl, 'describe': _describe_qwen2vl},
    'qwen35':  {'load': _load_qwen35,  'describe': _describe_qwen35},
}


def select_backend():
    """根据可用显存自动选择后端。"""
    vram = _get_free_vram_gb()
    if vram >= 35:
        name = "qwen35"
    elif vram >= 20:
        name = "qwen2vl"
    elif vram >= 16:
        name = "cogvlm"
    else:
        name = "blip"
    print(f"Auto-selected {name} (available VRAM: {vram:.1f}GB)")
    return name


def describe_frames(frame_paths, output_path, backend="auto", prompt=None, model_pair=None):
    """Generate text descriptions for a list of frame images.

    Args:
        frame_paths: List of image file paths.
        output_path: Where to save descriptions (one per line).
        backend: "auto", "blip", "cogvlm", or "qwen2vl".
        prompt: Custom prompt. If None, uses backend default.
        model_pair: (model, processor/tokenizer) tuple. If provided, reuse instead of loading.

    Returns:
        List of description strings.
    """
    # Check if already generated
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, 'r') as f:
            existing = f.readlines()
        if len(existing) == len(frame_paths):
            print(f"Descriptions already exist at {output_path} ({len(existing)} lines)")
            return [l.strip() for l in existing]

    if backend == "auto":
        backend = select_backend()

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Use: {', '.join(BACKENDS.keys())}")

    # 加载或复用模型
    should_unload = False
    if model_pair is None:
        model, proc = BACKENDS[backend]['load']()
        should_unload = True
    else:
        model, proc = model_pair

    descriptions = BACKENDS[backend]['describe'](frame_paths, model, proc, prompt=prompt)

    if should_unload:
        _unload(model, proc)

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for desc in descriptions:
            f.write(desc.strip() + '\n')
    print(f"Saved {len(descriptions)} descriptions to {output_path}")

    return descriptions


def describe_dataset(data_name, backend="auto", prompt=None):
    """为数据集的所有测试视频生成帧描述。模型只加载一次。

    用法:
        python vlm.py --data Mydataset
        python vlm.py --data Mydataset --backend qwen2vl
    """
    from glob import glob

    test_dir = os.path.join(data_name, 'test')
    desc_dir = os.path.join(data_name, 'test_frame_description')
    os.makedirs(desc_dir, exist_ok=True)

    video_dirs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    if not video_dirs:
        print(f"[错误] {test_dir} 下没有视频目录")
        return

    if backend == "auto":
        backend = select_backend()

    # 加载一次模型，处理所有视频
    print(f"共 {len(video_dirs)} 个测试视频待处理")
    model, proc = BACKENDS[backend]['load']()

    for i, vname in enumerate(video_dirs):
        frames = sorted(glob(os.path.join(test_dir, vname, '*.jpg')))
        output_path = os.path.join(desc_dir, f'{vname}.txt')
        print(f"\n[{i+1}/{len(video_dirs)}] {vname}: {len(frames)} 帧")
        describe_frames(frames, output_path, backend=backend, prompt=prompt, model_pair=(model, proc))

    _unload(model, proc)
    print(f"\n全部完成，描述文件保存在 {desc_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='为测试集生成帧描述')
    parser.add_argument('--data', type=str, required=True, help='数据集名称')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'blip', 'cogvlm', 'qwen2vl', 'qwen35'])
    parser.add_argument('--prompt', type=str, default=None, help='自定义提示词')
    args = parser.parse_args()
    describe_dataset(args.data, backend=args.backend, prompt=args.prompt)
