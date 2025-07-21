## 大概安装流程

### 步骤1: 创建环境并安装依赖

假设你现在位于PruLong目录下，用下面这些版本

```bash
# 创建conda环境
conda create -n prulong python=3.11
conda activate prulong

# cuda 12.4
conda install -y git
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y nvidia::cuda-cudart-dev
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
接下来，一定要注释掉requirements.txt文件里的block-sparse-attention包，然后自己从源码编译！！！!

而且requirements.txt里有些包会对于torch版本的依赖是互斥的，先注释掉mosaicml，之后有需要再安装

```bash
# 进入PruLong目录并安装依赖
cd prulong
python -m pip install -r requirements.txt # python -m保证用环境中的pip版本

# 安装Flash Attention
python -m pip install ninja packaging
python -m pip install flash-attn==2.6.1 --no-build-isolation

# Install Block Sparse Streaming Attention
cd .. # 回到PruLong目录
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention

python setup.py install
```

### 步骤2: 下载训练数据

```bash
cd ..  # 从Block Sparse Attention目录回到主目录

# 创建数据目录
mkdir -p datasets
cd datasets

# 下载预处理好的训练数据（很大，上百GB）
git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-64K long-context-65536

# 下载少量样本数据用于快速测试，这个数据集在huggingface上被删了
# git clone https://huggingface.co/datasets/princeton-nlp/prolong-sample prolong-sample

cd ../prulong  # 回到PruLong/prulong目录
```

### 步骤3: 开始训练

```bash
# 训练PruLong模型（仅训练掩码，适合已调优的模型）
# 这个是我改过的脚本，针对A100 8卡训练可行，只需要把模型路径换成自己的
bash run_scripts/prulong_masksonly.sh

```

完成！现在已经在成功训练PruLong了

训练过程中会看到：
- 损失函数下降
- 注意力头的稀疏性逐渐增加
- 训练进度和性能指标

## 自定义配置（可选）

### 修改训练参数
```bash
# 设置环境变量来自定义训练
export MODEL="meta-llama/Llama-3.1-8B-Instruct"  # 基础模型
export BSZ=8                                       # 批次大小（如果内存不够）
export STEPS=500                                   # 训练步数（快速测试）
export LR=1e-5                                     # 学习率

# 运行训练
bash run_scripts/prulong_masksonly.sh
```

### 选择训练模式

不确定下面2 3这两个个脚本跑得通

```bash
# 模式1：仅训练掩码（推荐，适合指令模型）
bash run_scripts/prulong_masksonly.sh

# 模式2：同时训练掩码和权重（更全面但可能影响已有调优）
bash run_scripts/prulong_masksandweights.sh

# 模式3：仅训练权重（适合已有PruLong模型的微调）
bash run_scripts/sft.sh
```

## 保存和使用结果

### 训练完成后保存掩码
```bash
# 训练完成后，保存学习到的掩码，其实在上面的run_scripts里会自动进行，所以不用着急手动调用
python save_prulong_masks.py --checkpoint checkpoints/你的检查点目录 --sparsity 0.7

# 这会生成 masks_sp0.7.tsv 文件，包含学习到的注意力头掩码
```

### 使用训练好的模型
```bash
# 掩码文件可以用于推理时的KV缓存优化
# 具体使用方法参考 eval/ 目录下的评估代码
```
