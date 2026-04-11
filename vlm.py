"""
VLM (Vision Language Model) module for generating frame-level scene descriptions.

Supports multiple backends:
  - CogVLM: High quality descriptions (~15GB VRAM), best for action/activity recognition
  - BLIP:   Lightweight alternative (~1.5GB VRAM fp16), better for scene/object description

Usage:
    from vlm import describe_frames

    # Auto-select based on available VRAM (default)
    descriptions = describe_frames(frame_paths, output_path)

    # Force a specific backend
    descriptions = describe_frames(frame_paths, output_path, backend="blip")
    descriptions = describe_frames(frame_paths, output_path, backend="cogvlm")
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


def _describe_blip(frame_paths, prompt="a surveillance camera image showing", batch_size=4):
    """Generate descriptions using BLIP-large (~1.5GB VRAM in fp16).

    Good for: scene/object-level description, low VRAM environments.
    Weak at: describing human actions, subtle motion, domain-specific objects.
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration

    blip_path = os.path.join(os.path.dirname(__file__), "blip-image-captioning-base")
    print(f"Loading BLIP model from {blip_path}...")
    processor = BlipProcessor.from_pretrained(blip_path)
    model = BlipForConditionalGeneration.from_pretrained(
        blip_path,
        torch_dtype=torch.float16
    ).to(device).eval()

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

    del model, processor
    torch.cuda.empty_cache()
    return descriptions


def _describe_cogvlm(frame_paths, prompt="Describe this image in detail."):
    """Generate descriptions using CogVLM (~15GB VRAM in bfloat16).

    Good for: detailed scene understanding, action recognition, object relationships.
    Requires: large VRAM (16GB+).
    """
    from modelscope import AutoModelForCausalLM, AutoTokenizer

    cogvlm_path = os.path.join(os.path.dirname(__file__), "cogvlm-chat")
    vicuna_path = os.path.join(os.path.dirname(__file__), "vicuna-7b-v1.5")
    print(f"Loading CogVLM model from {cogvlm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(vicuna_path)
    model = AutoModelForCausalLM.from_pretrained(
        cogvlm_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

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
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                images=images_tensor,
                max_new_tokens=512,
                do_sample=False,
            )
        resp = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        descriptions.append(resp.strip())

        if idx % 10 == 0 or idx == len(frame_paths) - 1:
            print(f"  [{idx + 1}/{len(frame_paths)}] {resp[:80]}...")

    del model, tokenizer
    torch.cuda.empty_cache()
    return descriptions


def describe_frames(frame_paths, output_path, backend="auto", prompt=None):
    """Generate text descriptions for a list of frame images.

    Args:
        frame_paths: List of image file paths.
        output_path: Where to save descriptions (one per line).
        backend: "auto" (select by VRAM), "blip", or "cogvlm".
        prompt: Custom prompt. If None, uses backend default.

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

    # Auto-select backend
    if backend == "auto":
        vram = _get_free_vram_gb()
        if vram >= 16:
            backend = "cogvlm"
            print(f"Auto-selected CogVLM (available VRAM: {vram:.1f}GB)")
        else:
            backend = "blip"
            print(f"Auto-selected BLIP (available VRAM: {vram:.1f}GB, need 16GB+ for CogVLM)")

    # Generate descriptions
    if backend == "blip":
        descriptions = _describe_blip(frame_paths, prompt=prompt or "a photo of")
    elif backend == "cogvlm":
        descriptions = _describe_cogvlm(frame_paths, prompt=prompt or "Describe this image in detail.")
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'blip', or 'cogvlm'.")

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for desc in descriptions:
            f.write(desc.strip() + '\n')
    print(f"Saved {len(descriptions)} descriptions to {output_path}")

    return descriptions


def describe_dataset(data_name, backend="auto", prompt=None):
    """为数据集的所有测试视频生成帧描述。

    自动扫描 {data_name}/test/ 下的所有视频目录，
    为每个视频生成 {data_name}/test_frame_description/{video_name}.txt。

    Args:
        data_name: 数据集名称（如 'Mydataset'）
        backend: "auto", "blip", 或 "cogvlm"
        prompt: 自定义提示词，None 则用默认值

    用法:
        python vlm.py --data Mydataset
        python vlm.py --data Mydataset --backend blip
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

    print(f"共 {len(video_dirs)} 个测试视频待处理")
    for i, vname in enumerate(video_dirs):
        frames = sorted(glob(os.path.join(test_dir, vname, '*.jpg')))
        output_path = os.path.join(desc_dir, f'{vname}.txt')
        print(f"\n[{i+1}/{len(video_dirs)}] {vname}: {len(frames)} 帧")
        describe_frames(frames, output_path, backend=backend, prompt=prompt)

    print(f"\n全部完成，描述文件保存在 {desc_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='为测试集生成帧描述')
    parser.add_argument('--data', type=str, required=True, help='数据集名称')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'blip', 'cogvlm'])
    parser.add_argument('--prompt', type=str, default=None, help='自定义提示词')
    args = parser.parse_args()
    describe_dataset(args.data, backend=args.backend, prompt=args.prompt)
