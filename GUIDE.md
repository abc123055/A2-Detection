# AnomalyRuler 使用指南

## 一、整体流程概述

AnomalyRuler 是一个基于规则的视频异常检测框架，不训练分类器，而是利用大模型将视觉信息转化为文字，再通过语言模型基于规则进行推理判断。

完整流程分为五个阶段：

```
视频 → 抽帧 → VLM 看图说话 → 关键词匹配+时序平滑 → LLM 规则推理 → 异常判断
        ↑          ↑                 ↑                    ↑
     dataset.py  vlm.py       majority_smooth.py       llm.py
```

| 阶段 | 文件 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| 1. 数据准备 | `dataset.py` | 视频/图片目录 | `train.csv`, `test_frame/*.csv` | 抽帧、生成路径+标签索引 |
| 2. 帧描述 | `vlm.py` | 图片帧 | `test_frame_description/*.txt` | VLM 为每帧生成文字描述 |
| 3. 规则归纳 | `llm.py` (`gpt_induction`) | 正常帧描述 | `rule/rule_XXX.txt` | GPT-4 从正常描述中归纳规则 |
| 4. 关键词+平滑 | `majority_smooth.py` | 描述 + 规则关键词 | `modified_test_frame_description/*.txt` | 关键词快筛 + EMA 时序平滑 |
| 5. LLM 推理 | `llm.py` (`mixtral_double_deduct`) | 修正后描述 + 规则 | `results/XXX/*.csv` | LLM 逐帧推理是否异常 |

---

## 二、环境配置

```bash
# 创建环境
conda env create -f environment.yml
conda activate env_ruler
```

核心依赖：Python 3.10, PyTorch 2.1.0 (CUDA 12.1), transformers 4.35.0, openai, scikit-learn, opencv-python

### 显存需求

| 模型 | 用途 | 显存 |
|------|------|------|
| BLIP-large | 图片描述（轻量） | ~1.5GB |
| CogVLM | 图片描述（高质量） | ~15GB |
| Mistral-7B (fp16) | 规则推理 | ~14GB |
| BART-Large-MNLI | 输出分类 | ~1.6GB |

> 注意：BLIP + Mistral-7B 无法同时加载在 6GB 显存的显卡上，流程是分步执行的（先跑完 VLM 释放显存，再加载 LLM）。

---

## 三、如何运行代码

### 3.1 数据目录结构

以 `SHTech` 为例，需要的目录结构如下：

```
SHTech/
├── train/                          # 正常视频的帧图片
│   └── video_01/
│       ├── frame_0000.jpg
│       └── ...
├── train.csv                       # 训练集索引 (image_path, label)，label 全为 0
├── test_frame/                     # 测试集标签文件
│   └── video_01.csv                # 列: image_path, label (0=正常, 1=异常)
├── test_frame_description/         # VLM 生成的原始帧描述 (每行一帧)
│   └── video_01.txt
└── modified_test_frame_description/ # 平滑后的帧描述 (majority_smooth.py 生成)
    └── video_01.txt
```

### 3.2 完整运行步骤

```bash
# 步骤 1: 生成帧描述（需要 VLM 模型）
# 见 vlm.py 的使用方式，或者用原始项目的 image2text.py

# 步骤 2: 归纳规则（需要 OpenAI API key）
# 在 llm.py 第 15 行填入 API key
python main.py --data='SHTech' --induct --b=10 --bs=1
# --b=10: 10 批次，每批采样不同的正常帧
# --bs=1: 每批采样 1 张帧
# 输出: 打印生成的规则，需手动保存到 rule/rule_SHTech.txt

# 步骤 3: 关键词匹配 + 时序平滑
python majority_smooth.py --data='SHTech'
# 输入: test_frame_description/*.txt + rule/rule_SHTech.txt
# 输出: modified_test_frame_description/*.txt

# 步骤 4: LLM 推理
python main.py --data='SHTech' --deduct
# 输入: modified_test_frame_description/*.txt + rule/rule_SHTech.txt
# 输出: results/SHTech/*.csv，并打印 ACC/Precision/Recall

# (可选) GPT-4 推理 demo（仅测试一个文件的前几帧）
python main.py --data='SHTech' --gpt_deduct_demo
```

### 3.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `SHTech` | 数据集名称，同时也是数据目录名 |
| `--induct` | false | 执行规则归纳阶段 |
| `--deduct` | false | 执行 LLM 推理阶段 |
| `--gpt_deduct_demo` | false | 用 GPT-4 API 跑推理 demo |
| `--b` | 10 | 规则归纳的批次数 |
| `--bs` | 1 | 每批采样帧数 |

---

## 四、如何换数据集

### 4.1 使用已支持的数据集

代码内置支持 4 个数据集：`SHTech`、`avenue`、`ped2`、`UBNormal`

```bash
python main.py --data='avenue' --deduct
```

对应关系在 `main.py` 第 25 行：
```python
data_full_name = {'SHTech':'ShanghaiTech', 'avenue':'CUHK Avenue', 'ped2': 'UCSD Ped2', 'UBNormal': 'UBNormal'}[data_name]
```

每个数据集需要：
1. 同名目录下的数据文件（见 3.1 的目录结构）
2. `rule/rule_XXX.txt` 规则文件
3. `majority_smooth.py` 中对应的关键词列表或 `rule/rule_XXX.npy`

### 4.2 添加新数据集

以添加名为 `wafer` 的晶圆数据集为例：

**第一步：修改 `main.py`**

```python
# 第 12 行，添加 choices
parser.add_argument('--data', type=str, default='SHTech',
                    choices=['SHTech', 'avenue', 'ped2', 'UBNormal', 'wafer'])

# 第 25 行，添加映射
data_full_name = {
    'SHTech':'ShanghaiTech', 'avenue':'CUHK Avenue',
    'ped2': 'UCSD Ped2', 'UBNormal': 'UBNormal',
    'wafer': 'Wafer Manufacturing'   # <-- 新增
}[data_name]
```

**第二步：准备数据目录**

```
wafer/
├── train/                          # 放正常视频的帧
├── train.csv                       # 用 dataset.py 生成
├── test_frame/
│   └── test_video_01.csv           # 每帧的标签
├── test_frame_description/         # VLM 生成的描述
│   └── test_video_01.txt
└── modified_test_frame_description/ # majority_smooth.py 生成
    └── test_video_01.txt
```

生成 `train.csv`：
```python
from dataset import create_train_csv
create_train_csv('wafer')
```

**第三步：编写规则文件 `rule/rule_wafer.txt`**（可以手写，也可以用 `--induct` 自动生成）

**第四步：配置关键词列表**

在 `majority_smooth.py` 中修改 `anomaly_keywords()` 函数，或直接生成 `rule/rule_wafer.npy`：

```python
import numpy as np
keywords = ["水珠", "晃动", "多滴", ...]  # 你的异常关键词
np.save('rule/rule_wafer.npy', keywords)
```

> 注意：`majority_smooth.py` 第 160 和 195-198 行硬编码了 `rule/rule_SHTech.npy` 和 `rule/rule_SHTech.txt`，换数据集时需要改为对应路径，或改为根据 `--data` 参数动态拼接。

**第五步：运行**

```bash
python majority_smooth.py --data='wafer'
python main.py --data='wafer' --deduct
```

---

## 五、如何换模型

### 5.1 换 VLM（图片描述模型）

VLM 在 `vlm.py` 中实现，当前支持两个后端：

- `blip`：Salesforce/blip-image-captioning-large（~1.5GB 显存）
- `cogvlm`：ZhipuAI/cogvlm-chat（~15GB 显存）

**添加新 VLM 后端的步骤：**

1. 在 `vlm.py` 中新增一个 `_describe_xxx()` 函数
2. 在 `describe_frames()` 中添加对应的分支

示例——添加 Qwen2-VL-2B 后端：

```python
# vlm.py 新增函数
def _describe_qwen2vl(frame_paths, prompt="请描述这张图片中的场景"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    descriptions = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=256)
        desc = processor.batch_decode(output, skip_special_tokens=True)[0]
        descriptions.append(desc)
    
    del model, processor
    torch.cuda.empty_cache()
    return descriptions

# 在 describe_frames() 的分支中添加:
elif backend == "qwen2vl":
    descriptions = _describe_qwen2vl(frame_paths, prompt=prompt or "描述这张图片")
```

### 5.2 换 LLM（规则推理模型）

LLM 在 `main.py` 中加载，在 `llm.py` 中使用。当前默认为 Mistral-7B-Instruct-v0.2。

**修改位置：`main.py` 第 62-64 行**

```python
# 原始代码
model_id = "AI-ModelScope/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id, ...)
```

换成其他模型只需改 `model_id`，例如：

```python
# TinyLlama 1.1B（显存更小，~2.2GB fp16）
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Qwen2-1.5B-Instruct
model_id = "Qwen/Qwen2-1.5B-Instruct"

# Llama-3-8B-Instruct（需要更大显存）
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
```

> 注意：不同模型的 chat template 可能不同。Mistral 使用 `[INST]...[/INST]` 格式，当前代码是直接拼接文本作为 prompt（没有用 chat template），所以大部分 Instruct 模型可以直接替换。但如果推理效果不好，可能需要在 `llm.py` 的 `mixtral_deduct` 和 `mixtral_double_deduct` 函数中调整 prompt 格式。

### 5.3 换归纳阶段的 GPT 模型

`llm.py` 中 `gpt_induction` 和 `gpt_rule_correction` 函数使用 OpenAI API。

**修改位置：`llm.py` 第 33 行和第 69 行**

```python
model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4"]
model = model_list[3]  # 当前用 gpt-4
```

可以改为 `"gpt-4o"` 或 `"gpt-4o-mini"` 等。

### 5.4 换后处理分类器

`utils.py` 第 14 行加载了 BART-Large-MNLI 用于零样本分类：

```python
classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
```

这个分类器用于从 LLM 的推理输出中提取最终的 normal/anomaly 标签（`post_process` 函数）。如果要换成更小的模型：

```python
# 更小的 DeBERTa 模型
classifier = hf_pipeline("zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)
```

---

## 六、如何使用本地模型

所有模型默认从 HuggingFace Hub 下载。如果需要使用本地已下载的模型，将 model ID 替换为本地路径即可。

### 6.1 下载模型到本地

```bash
# 方式一：用 huggingface-cli
huggingface-cli download AI-ModelScope/Mistral-7B-Instruct-v0.2 --local-dir /path/to/models/mistral-7b

# 方式二：用 git lfs
git lfs install
git clone https://huggingface.co/AI-ModelScope/Mistral-7B-Instruct-v0.2 /path/to/models/mistral-7b

# 方式三：用 modelscope（国内更快）
pip install modelscope
modelscope download --model AI-ModelScope/Mistral-7B-Instruct-v0.2 --local_dir /path/to/models/mistral-7b
```

### 6.2 使用本地模型

**LLM（`main.py`）：**
```python
model_id = "/path/to/models/mistral-7b"  # 改为本地路径
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id, ...)
```

**VLM - CogVLM（`vlm.py`）：**
```python
tokenizer = AutoTokenizer.from_pretrained('/path/to/models/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained('/path/to/models/cogvlm-chat', ...)
```

**VLM - BLIP（`vlm.py`）：**
```python
processor = BlipProcessor.from_pretrained('/path/to/models/blip-image-captioning-large')
model = BlipForConditionalGeneration.from_pretrained('/path/to/models/blip-image-captioning-large', ...)
```

**后处理分类器（`utils.py`）：**
```python
classifier = hf_pipeline("zero-shot-classification",
    model="/path/to/models/bart-large-mnli", device=device)
```

### 6.3 离线运行

设置环境变量禁止联网下载：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

此时所有 `from_pretrained()` 调用必须指向本地路径，否则会报错。

---

## 七、适配晶圆场景

### 7.1 总体思路

将 AnomalyRuler 从监控视频（行人/街道）适配到晶圆制造场景，核心改动在三处：
1. **规则文件**：从"行人异常"改为"制造异常"
2. **关键词列表**：从"riding/running/bicycle"改为晶圆领域关键词
3. **VLM prompt**：引导模型关注工业设备而非行人

### 7.2 编写晶圆规则文件

创建 `rule/rule_wafer.txt`，保持与原有规则文件相同的格式：

```
**Rules for Anomaly Human Activities:**
1. N/A (晶圆场景通常无人活动相关异常)

**Rules for Anomaly Environmental Objects:**
1. Dropper dispensing extra drops or missing drops during adhesive application
2. Unexpected liquid droplets or water beads on equipment surface
3. Wafer base/stage exhibiting abnormal vibration or displacement
4. Adhesive line trajectory deviation, breakage, or uneven thickness
5. Equipment components loose or misaligned
6. Abnormal color or texture on wafer surface
7. Foreign particles or contamination on wafer

**Rules for Normal Human Activities:**
1. N/A

**Rules for Normal Environmental Objects:**
1. Dropper dispensing single drop of adhesive per cycle, with even flow
2. Wafer base/stage stable and stationary
3. Equipment surface dry and clean
4. All components in correct position and properly secured
5. Wafer surface uniform in color and texture
6. Adhesive line smooth, continuous, and even in thickness
```

### 7.3 配置晶圆关键词列表

关键词用于 `majority_smooth.py` 的快速筛查阶段。

```python
import numpy as np
wafer_anomaly_keywords = [
    # 英文关键词（如果 VLM 输出英文描述）
    "droplet", "drip", "splash", "leak", "vibration", "shaking",
    "wobble", "misaligned", "crack", "scratch", "contamination",
    "particle", "bubble", "overflow", "deviation", "broken",
    "loose", "displaced", "stain", "residue",
]
np.save('rule/rule_wafer.npy', wafer_anomaly_keywords)
```

### 7.4 需要修改的提示词位置

适配晶圆场景需要修改以下提示词。下面按流程阶段逐一说明每个 prompt 对应的**模型是什么、模型在流程中的作用是什么、这条 prompt 起什么效果、以及如何改**。

---

#### (1) VLM 图片描述 prompt — 整个流程的信息源头

> **对应模型**：BLIP-large（`Salesforce/blip-image-captioning-large`）或 CogVLM（`ZhipuAI/cogvlm-chat`）
>
> **模型作用**：VLM（Vision Language Model，视觉语言模型）接收一张图片，输出一句文字描述。它是整个流程的第一步，负责把视觉信息"翻译"成自然语言。后续所有阶段（关键词匹配、LLM 推理）都建立在这个描述之上，因此**描述质量直接决定最终检测效果**。
>
> **prompt 的作用**：引导 VLM 关注图片中的哪些内容。如果 prompt 说"surveillance camera"，模型会倾向于描述行人和街道；如果改成"wafer manufacturing"，模型会尝试关注工业设备和制造过程。这是适配新场景最关键的修改点。

**文件：`vlm.py`**

**改动 1：`_describe_blip()` 第 35 行** — BLIP 后端的默认 prompt

BLIP 是一个轻量级的图像描述模型（~1.5GB 显存），采用 encoder-decoder 架构。它的 prompt 作为文本前缀（prefix），模型在此基础上续写描述。prompt 越贴合场景，生成的描述越有针对性。

```python
# 原始：引导模型关注监控画面
prompt="a surveillance camera image showing"
# 改为：引导模型关注晶圆制造设备
prompt="a manufacturing camera image showing wafer processing equipment"
```

**改动 2：`_describe_cogvlm()` 第 75 行** — CogVLM 后端的默认 prompt

CogVLM 是一个更大的多模态对话模型（~15GB 显存），理解能力更强。它的 prompt 是一个完整的问题/指令，模型以对话方式回答。可以在 prompt 中明确列出需要关注的设备部件，让描述包含更多有用信息。

```python
# 原始：通用描述指令
prompt="Describe this image in detail."
# 改为：指定关注晶圆制造中的关键部件和状态
prompt="Describe this wafer manufacturing image in detail, focusing on the dropper, adhesive, wafer surface, and equipment status."
```

**改动 3：`describe_frames()` 第 155-157 行** — 统一入口的 fallback 默认值

`describe_frames()` 是对外暴露的统一接口。当调用时不传 `prompt` 参数，会使用这里的默认值。这里的修改保证即使不显式传 prompt，也会使用晶圆场景的描述指令。

```python
# 原始
prompt=prompt or "describe this image"                      # blip 分支
prompt=prompt or "Describe this image in detail."           # cogvlm 分支
# 改为
prompt=prompt or "describe the wafer manufacturing equipment status"
prompt=prompt or "Describe this wafer manufacturing image in detail, focusing on equipment state and any anomalies."
```

---

#### (2) 规则归纳 prompt — 从正常描述中自动生成检测规则

> **对应模型**：GPT-4（通过 OpenAI API 调用）
>
> **模型作用**：GPT-4 在归纳阶段扮演"规则总结者"的角色。它接收一批正常帧的 VLM 描述，总结出"什么样的场景是正常的"和"什么样的场景可能是异常的"，输出结构化的规则列表。这些规则保存到 `rule/rule_XXX.txt`，供后续推理阶段使用。
>
> **prompt 的作用**：通过 system prompt 设定 GPT-4 的"身份"和"任务领域"，通过 user prompt 和 assistant 示例格式引导它按照特定分类维度（人类活动 vs 环境物体，或设备行为 vs 产品状态）来组织规则。如果不修改这些 prompt，GPT-4 会按照"城市安全监控"的思路去总结，生成的规则全是关于行人和交通的。

**文件：`llm.py`**

**改动 4：`gpt_induction()` 第 39 行** — system prompt（设定模型身份）

这条 prompt 告诉 GPT-4"你是谁、你在做什么"。它决定了模型理解后续所有输入的上下文框架。

```python
# 原始：身份是城市安全监控员
f"As a surveillance monitor for urban safety using the {data_full_name} dataset, my job is derive rules for detect abnormal human activities or environmental object."
# 改为：身份是晶圆制造质检员
f"As a quality inspector for wafer manufacturing using the {data_full_name} dataset, my job is to derive rules for detecting abnormal equipment behavior or product defects."
```

**改动 5：`gpt_induction()` 第 41 行** — user prompt（指导归纳方向）

这条 prompt 告诉 GPT-4 如何从正常描述中归纳规则。括号中的举例非常重要——它暗示了模型应该往哪个方向思考。原始示例是"walking"，模型就会围绕行走展开；改成晶圆场景的示例，模型会围绕制造过程展开。

```python
# 原始：示例是行走活动
"Based on the assumption that the given frame description are normal, Please derive rules for normal, start from an abstract concept and then generalize to concrete activities (e.g., walking) or objects."
# 改为：示例是晶圆制造行为
"Based on the assumption that the given frame descriptions are from normal manufacturing process, please derive rules for normal equipment behavior and product state, from abstract concepts to concrete observations (e.g., single drop per cycle, stable stage)."
```

**改动 6：`gpt_induction()` 第 43-48 行** — assistant 示例格式（规定输出结构）

这是 few-shot 中的 assistant 回复模板，它规定了 GPT-4 输出规则时使用的分类维度和格式。原始维度是"Human Activities / Environmental Objects"，这是监控场景的分类方式。晶圆场景应改为"Equipment Behavior / Product State"。

```python
# 原始：按人类活动和环境物体分类
'''**Rules for Normal Human Activities:
  1.
  **Rules for Normal Environmental Objects:
  1. '''
# 改为：按设备行为和产品状态分类
'''**Rules for Normal Equipment Behavior:
  1.
  **Rules for Normal Product State:
  1. '''
```

**改动 7：`gpt_induction()` 第 50 行** — 异常规则请求（引导生成异常规则）

这条 prompt 让 GPT-4 在正常规则的基础上"反推"异常规则。括号中的示例同样起引导作用——告诉模型异常应该往什么方向想。

```python
# 原始：异常示例是骑车、滑板等
"Compared with the above rules for normal, can you provide potential rules for anomaly? Please start from an abstract concept then generalize to concrete activities (e.g., non-walking such as riding a bicycle, scooting, skateboarding.) or objects, compared with normal ones."
# 改为：异常示例是多滴、污染、振动等
"Compared with the above rules for normal, can you provide potential rules for anomaly? Please start from abstract concepts then generalize to concrete defects (e.g., extra drops, surface contamination, stage vibration) compared with normal ones."
```

**改动 8：`gpt_rule_correction()` 第 75 行** — 规则整合的 system prompt

`gpt_rule_correction()` 在多轮归纳之后调用，作用是把多批次独立生成的规则合并整理成一份统一的规则集。它的 system prompt 也需要从"城市安全"改为"晶圆制造"。

```python
# 原始
f"As a surveillance monitor for urban safety using the {data_full_name} dataset, my job is to organize rules for detect abnormal activities and objeacts."
# 改为
f"As a quality inspector for wafer manufacturing using the {data_full_name} dataset, my job is to organize rules for detecting abnormal equipment behavior and product defects."
```

---

#### (3) LLM 推理 prompt — 逐帧判断是否异常的核心逻辑

> **对应模型**：Mistral-7B-Instruct-v0.2（本地运行，`AI-ModelScope/Mistral-7B-Instruct-v0.2`）
>
> **模型作用**：Mistral-7B 是推理阶段的核心模型。对于每一帧的 VLM 描述，它接收描述文本和规则文件，按照 prompt 中设定的推理步骤（chain-of-thought）逐步分析：先匹配规则，再给出判断，最后输出置信度。它的输出会被 `post_process()` 解析，提取最终的 normal/anomaly 标签。
>
> **prompt 的作用**：这是整个推理链的"指令模板"。它定义了模型的推理步骤（First/Second/Third/Fourth），以及每一步要关注什么维度。如果 prompt 仍然要求模型检查"human activity"和"non-human object"，那么即使规则文件已经改成晶圆内容，模型的推理过程也会不匹配。

**文件：`llm.py`**

**改动 9：`mixtral_deduct()` 第 111-117 行** — 单次推理的完整 prompt

这是最核心的推理 prompt。它包含四个推理步骤，引导 Mistral-7B 先分维度匹配规则，再综合判断，最后给出置信度。每个步骤中提到的分析维度必须与规则文件中的分类维度一致。

```python
# 原始：按"人类活动"和"非人类物体"两个维度推理
text = f'''You will be given an description of scene, you task is to detect the anomaly based on the rules. The rules are:
    {rule}\n\n
    First, if human activity present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
    Second, if non-human object present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
    Third, are the human activities or non-human objects anomaly? Answer: anomaly, if ANY anomaly rule (even if only one, no matter human activities or non-human objects) matches, otherwise answer: normal.\n\n
    Fourth, describe how likely it is that your best answer is correct as one of the following expressions: ...'''
# 改为：按"设备行为"和"产品状态"两个维度推理
text = f'''You will be given a description of a wafer manufacturing scene. Your task is to detect anomalies based on the rules. The rules are:
    {rule}\n\n
    First, check the equipment behavior: which rule is matching? List the rule category (normal or anomaly) with number.\n\n
    Second, check the product state: which rule is matching? List the rule category (normal or anomaly) with number.\n\n
    Third, is the equipment behavior or product state anomalous? Answer: anomaly, if ANY anomaly rule (even if only one, whether equipment behavior or product state) matches, otherwise answer: normal.\n\n
    Fourth, describe how likely it is that your best answer is correct as one of the following expressions: ...'''
```

**改动 10：`mixtral_double_deduct()` 第 169-185 行** — 二次确认推理的 prompt

这个函数实现"双重推理"：先用关键词匹配做初步判断（`cluster_keyword`），再用 Mistral-7B 对初步结果做二次确认。prompt 中会告诉模型"我的初步判断是 XXX，请你核实"。当初步判断为异常时，prompt 会额外提示模型关注触发异常的关键词；当初步判断为正常时，prompt 仅要求模型独立验证。

需要修改的内容与改动 9 相同：
- 将 `"human activity"` → `"equipment behavior"`
- 将 `"environmental object"` → `"product state"`

```python
# 异常分支（ini_pred == 1）的 prompt
# 原始
text = f'''... my initial result is {ini_answer}\n
    First, if human activity present, which rule is matching? ...
    Second, if environmental object present, which rule is matching? ...
    Third, are the human activities or environmental objects anomaly? Answer: anomaly, if you also find {anomaly_keyword[0]} or ANY anomaly rule ...'''
# 改为
text = f'''... my initial result is {ini_answer}\n
    First, check the equipment behavior: which rule is matching? ...
    Second, check the product state: which rule is matching? ...
    Third, is the equipment behavior or product state anomalous? Answer: anomaly, if you also find {anomaly_keyword[0]} or ANY anomaly rule ...'''

# 正常分支（ini_pred == 0）的 prompt 做同样修改
```

**改动 11：`gpt_double_deduction_demo()` 第 222-248 行** — GPT-4 版本的二次推理 demo

> **对应模型**：GPT-4（通过 OpenAI API）

这个函数的逻辑与 `mixtral_double_deduct()` 完全相同，区别是用 GPT-4 API 代替本地 Mistral-7B。它额外有一个 system prompt（第 243 行）需要修改：

```python
# 原始
f"As a surveillance monitor for urban safety using the {data} dataset, my job to detect abnormal human activities or environmental object based on the provided rules"
# 改为
f"As a quality inspector for wafer manufacturing using the {data} dataset, my job is to detect abnormal equipment behavior or product defects based on the provided rules"
```

---

#### (4) 后处理分类标签 — 从 LLM 输出中提取最终判断

> **对应模型**：BART-Large-MNLI（`facebook/bart-large-mnli`）
>
> **模型作用**：BART-Large-MNLI 是一个零样本文本分类器。LLM（Mistral-7B）的推理输出是一段自由文本（可能包含分析过程、匹配的规则编号、最终结论等），`post_process()` 函数取输出的最后一句话，用 BART 判断这句话更接近"normal"还是"anomaly"，从而提取出结构化的 0/1 标签。
>
> **prompt 的作用**：这里的"prompt"实际上是分类标签列表 `["normal", "anomaly"]`。BART 会计算输入文本与每个标签的语义相似度。

**文件：`utils.py`**

`post_process()` 第 193 行：
```python
result = classifier(last_sentence, candidate_labels=["normal", "anomaly"])
```

`"normal"` 和 `"anomaly"` 是通用的语义标签，不依赖具体场景，**无需修改**。无论 LLM 的输出是关于行人还是关于晶圆，BART 都能判断最后一句话的语义倾向。

---

#### 提示词修改清单总结

| 编号 | 文件 | 行号 | 对应模型 | 模型职责 | prompt 作用 |
|------|------|------|----------|----------|-------------|
| 1 | vlm.py | 35 | BLIP | 图片→文字描述 | 引导描述关注晶圆设备而非行人街道 |
| 2 | vlm.py | 75 | CogVLM | 图片→文字描述 | 同上，指定关注部件（滴管、晶圆面等） |
| 3 | vlm.py | 155-157 | BLIP/CogVLM | 图片→文字描述 | 统一入口的默认 prompt |
| 4 | llm.py | 39 | GPT-4 (API) | 归纳正常/异常规则 | 设定"质检员"身份，框定分析领域 |
| 5 | llm.py | 41 | GPT-4 (API) | 归纳正常/异常规则 | 引导从制造过程角度归纳规则 |
| 6 | llm.py | 43-48 | GPT-4 (API) | 归纳正常/异常规则 | 规定输出格式的分类维度 |
| 7 | llm.py | 50 | GPT-4 (API) | 归纳正常/异常规则 | 用制造缺陷示例引导异常规则方向 |
| 8 | llm.py | 75 | GPT-4 (API) | 整合多批规则 | 设定整合任务的领域上下文 |
| 9 | llm.py | 111-117 | Mistral-7B (本地) | 逐帧推理判断 | 定义推理步骤和分析维度 |
| 10 | llm.py | 169-185 | Mistral-7B (本地) | 二次确认推理 | 同上，加入初步判断的上下文 |
| 11 | llm.py | 222-248 | GPT-4 (API) | 二次推理 demo | 同 10，API 版本 |
| 12 | utils.py | 193 | BART-MNLI | 提取 0/1 标签 | 通用标签，无需修改 |

### 7.5 晶圆场景的局限性与应对

| 问题 | 原因 | 应对建议 |
|------|------|----------|
| VLM 不认识工业设备 | 通用 VLM 没见过晶圆生产线图片 | 换用更强的 VLM（如 Qwen2-VL-7B）或对小模型做 LoRA 微调 |
| 时序异常无法检测 | 多滴/漏滴/晃动是跨帧模式，逐帧描述无法捕捉 | 引入帧差分特征，或用视频理解模型替代逐帧描述 |
| 视觉差异极细微 | 水珠可能只有几个像素，VLM resize 后丢失 | 对感兴趣区域（ROI）裁剪放大后再描述 |
| 规则难定量 | "振动多大算晃动"无法用自然语言精确表达 | 配合传统 CV 方法（帧差分、光流）做定量判断 |

### 7.6 推荐的适配路线

1. **先跑通**：用 BLIP 或 CogVLM + 手写规则，跑通全流程，观察 VLM 描述质量
2. **调 prompt**：根据 VLM 实际输出，迭代调整描述 prompt 和关键词列表
3. **加帧差分**：在 VLM 描述之外，计算相邻帧像素差异作为补充特征
4. **考虑微调**：如果通用 VLM 描述质量始终不够，收集 200+ 标注图片对 VLM 做 LoRA 微调

---

## 八、代码结构速查

```
AnomalyRuler/
├── main.py                  # 主入口：--induct 归纳规则，--deduct 推理检测
├── vlm.py                   # VLM：BLIP / CogVLM 后端，生成帧描述
├── llm.py                   # LLM：GPT-4 归纳规则 + Mistral 推理判断
├── utils.py                 # 工具：BART 分类器、评估指标、文件读写
├── dataset.py               # 数据：抽帧、标签加载、CSV 生成
├── majority_smooth.py       # 关键词匹配 + EMA 时序平滑
├── openai_api.py            # OpenAI API：关键词提取、基线方法
├── generate_choices.py      # 生成选择题格式的训练样本
├── rule/                    # 规则文件目录
├── results/                 # 推理结果输出目录
├── environment.yml          # Conda 环境配置
└── run.sh                   # 运行脚本示例
```

---

## 九、关键参数调优

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `--b` | main.py | 10 | 规则归纳批次数 |
| `--bs` | main.py | 1 | 每批采样帧数 |
| `alpha` | majority_smooth.py:157 | 0.33 | EMA 平滑系数，越大越敏感 |
| `window_size` | majority_smooth.py:158 | 1 | 多数投票窗口大小 |
| `max_new_tokens` | llm.py:119 | 4000 | LLM 最大生成长度 |
| `threshold` | majority_smooth.py:157 | 动态计算 | EMA 均值作为异常阈值 |

晶圆场景可能需要调整：
- **`alpha`**：工业视频帧率和异常持续时间与监控视频不同，可能需要更小的值（更平滑）
- **`window_size`**：如果异常持续时间较长（如持续晃动），适当增大
- **`max_new_tokens`**：如果 LLM 输出被截断，适当增大
