# SMoE 说明文档

## 1. 建立文件夹
因为代码中都用的绝对路径，所以先建立几个文件夹
```bash
mkdir /mnt/ky2307909/going/data
mkdir /mnt/ky2307909/going/smoexversemoe-4060/ourwork
```
把项目放在目录/mnt/ky2307909/going/smoexversemoe-4060/ourwork下面

## 2. 模型准备

### 下载 Qwen2-57B-A14B-Instruct 模型
这里给出从 HF 镜像下载的指令：

```bash
git clone https://hf-mirror.com/Qwen/Qwen2-57B-A14B-Instruct ./Qwen2-57B-A14B-Instruct
```

下载后，找到项目下的 `div_tensors.py` 脚本。将第六行的模型目录改成你下载的 MoE 模型的目录。

## 3. 数据集下载

请使用以下 Python 脚本下载所需数据集。

```python
import os
import json
import requests
from tqdm import tqdm
from datasets import load_dataset

# ===================== 【核心配置 - 只改这里！】 =====================
# 你想把数据集下载到的根目录，比如 /home/xxx/data 或 D:/data
# 下载完成后生成的目录：ROOT_DATA_DIR/GAOKAO-BENCH/ 、ROOT_DATA_DIR/gsm8k/ 等
ROOT_DATA_DIR = "/mnt/ky2307909/going/data"
# ====================================================================

# 国内加速配置 - 解决HuggingFace下载慢的问题，必须加！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# 自动创建和原脚本完全一致的目录结构
dirs_to_create = [
    f"{ROOT_DATA_DIR}/GAOKAO-BENCH/data/Multiple-choice_Questions",
    f"{ROOT_DATA_DIR}/SuperGLUE/WiC",
    f"{ROOT_DATA_DIR}/triviaqa",
    f"{ROOT_DATA_DIR}/race/validation",
    f"{ROOT_DATA_DIR}/gsm8k"
]
for dir_path in dirs_to_create:
    os.makedirs(dir_path, exist_ok=True)
print(f"✅ 已创建所有目录，根路径：{ROOT_DATA_DIR}")

# -------------------------- 下载【1. 高考题库 GAOKAO-BENCH 4个文件】 --------------------------
GAOKAO_FILES = [
    "2010-2022_Math_I_MCQs.json",
    "2010-2022_Math_II_MCQs.json",
    "2010-2022_History_MCQs.json",
    "2010-2022_Biology_MCQs.json"
]
GAOKAO_BASE_URL = "https://raw.githubusercontent.com/OpenLMLab/GAOKAO-Bench/main/data/Multiple-choice_Questions/"
save_gaokao_dir = f"{ROOT_DATA_DIR}/GAOKAO-BENCH/data/Multiple-choice_Questions/"

print("\n===== 开始下载【高考题库 GAOKAO-BENCH】 =====")
for filename in GAOKAO_FILES:
    save_path = os.path.join(save_gaokao_dir, filename)
    if os.path.exists(save_path):
        print(f"✅ 跳过 {filename} (文件已存在)")
        continue
    try:
        url = GAOKAO_BASE_URL + filename
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        total_size = int(resp.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(desc=filename, total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        print(f"✅ 下载完成: {filename}")
    except Exception as e:
        print(f"❌ 下载 {filename} 失败: {str(e)}")

# -------------------------- 下载【2. SuperGLUE-WiC val.jsonl】 --------------------------
print("\n===== 开始下载【SuperGLUE-WiC】 =====")
wic_save_path = f"{ROOT_DATA_DIR}/SuperGLUE/WiC/val.jsonl"
if os.path.exists(wic_save_path):
    print(f"✅ 跳过 val.jsonl (文件已存在)")
else:
    try:
        dataset = load_dataset("super_glue", "wic", split="validation")
        with open(wic_save_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 下载完成: val.jsonl")
    except Exception as e:
        print(f"❌ WiC下载失败: {str(e)}")

# -------------------------- 下载【3. TriviaQA triviaqa-train.jsonl】 --------------------------
print("\n===== 开始下载【TriviaQA】 =====")
triviaqa_save_path = f"{ROOT_DATA_DIR}/triviaqa/triviaqa-train.jsonl"
if os.path.exists(triviaqa_save_path):
    print(f"✅ 跳过 triviaqa-train.jsonl (文件已存在)")
else:
    try:
        dataset = load_dataset("trivia_qa", "rc", split="train")
        with open(triviaqa_save_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 下载完成: triviaqa-train.jsonl")
    except Exception as e:
        print(f"❌ TriviaQA下载失败: {str(e)}")

# -------------------------- 下载【4. RACE 阅读理解 middle.jsonl + high.jsonl】 --------------------------
print("\n===== 开始下载【RACE 初中+高中】 =====")
race_mid_save = f"{ROOT_DATA_DIR}/race/validation/middle.jsonl"
race_high_save = f"{ROOT_DATA_DIR}/race/validation/high.jsonl"

if os.path.exists(race_mid_save):
    print(f"✅ 跳过 middle.jsonl (文件已存在)")
else:
    try:
        dataset = load_dataset("race", "middle", split="validation")
        with open(race_mid_save, "w", encoding="utf-8") as f:
            for sample in dataset:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 下载完成: middle.jsonl")
    except Exception as e:
        print(f"❌ RACE初中下载失败: {str(e)}")

if os.path.exists(race_high_save):
    print(f"✅ 跳过 high.jsonl (文件已存在)")
else:
    try:
        dataset = load_dataset("race", "high", split="validation")
        with open(race_high_save, "w", encoding="utf-8") as f:
            for sample in dataset:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 下载完成: high.jsonl")
    except Exception as e:
        print(f"❌ RACE高中下载失败: {str(e)}")

# -------------------------- 下载【5. GSM8K train.jsonl】 --------------------------
print("\n===== 开始下载【GSM8K 数学推理】 =====")
gsm8k_save_path = f"{ROOT_DATA_DIR}/gsm8k/train.jsonl"
if os.path.exists(gsm8k_save_path):
    print(f"✅ 跳过 train.jsonl (文件已存在)")
else:
    try:
        dataset = load_dataset("gsm8k", "main", split="train")
        with open(gsm8k_save_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 下载完成: train.jsonl")
    except Exception as e:
        print(f"❌ GSM8K下载失败: {str(e)}")

print("\n🎉🎉🎉 所有数据集下载任务执行完毕！")
print(f"📁 数据集根目录: {ROOT_DATA_DIR}")
print("✅ 你的原代码可以直接运行，无需修改任何路径！")
```

## 4. 激活环境

```bash
source /root/.bashrc
conda init
conda activate .vllm_ascend_latest/
```

> **注意**：这里的 `.vllm_ascend_latest` 文件夹不在项目仓库里，可能要单独提供。

## 5. 安装依赖

依次执行如下两个脚本，安装运行时所要用到的必要依赖。

### 5.1 系统依赖 (`install_apt_dependency.sh`)

```bash
# Using apt-get with mirror
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq fish autossh

# Or using yum
# yum update -y && yum install -y gcc g++ cmake numactl-devel wget git curl jq

# Config pip mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 5.2 Ascend 依赖 (`Install_ascend_dependency.sh`)

```bash
# sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
# apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq

chmod +x ./Ascend-cann-toolkit_8.3.RC1_linux-"$(uname -i)".run
./Ascend-cann-toolkit_8.3.RC1_linux-"$(uname -i)".run --full

chmod +x ./Ascend-cann-kernels-910b_8.3.RC1_linux-"$(uname -i)".run
./Ascend-cann-kernels-910b_8.3.RC1_linux-"$(uname -i)".run --install

chmod +x ./Ascend-cann-nnal_8.3.RC1_linux-"$(uname -i)".run
./Ascend-cann-nnal_8.3.RC1_linux-"$(uname -i)".run --install

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 6. 运行

在项目目录下执行：

```bash
python main.py
```
