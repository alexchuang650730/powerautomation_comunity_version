# SRT 依赖配置

## 1. 依赖包

```
srt-learning==0.8.5
rloo-algorithm==1.1.0
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.2
tqdm>=4.61.0
```

## 2. 环境变量配置

```bash
# SRT模型配置
SRT_MODEL_PATH=/path/to/models
SRT_DATA_PATH=/path/to/training_data
SRT_CONFIG_PATH=/path/to/config

# 训练配置
SRT_BATCH_SIZE=32
SRT_LEARNING_RATE=0.001
SRT_MAX_ITERATIONS=1000

# 资源配置
SRT_NUM_WORKERS=4
SRT_USE_GPU=true
SRT_GPU_DEVICE=0

# 日志配置
SRT_LOG_LEVEL=INFO
SRT_LOG_FILE=/var/log/srt-adapter.log
```

## 3. 资源需求

- **CPU**: 最低4核，推荐8核
- **GPU**: 推荐NVIDIA GPU，至少8GB显存
- **内存**: 最低8GB，推荐16GB
- **存储**: 最低20GB可用空间
- **CUDA**: 版本11.0或更高

## 4. 安装步骤

### 4.1 使用pip安装

```bash
# 创建虚拟环境
python -m venv srt-env
source srt-env/bin/activate

# 安装依赖
pip install -r requirements-srt.txt

# 安装CUDA支持（如果使用GPU）
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### 4.2 使用Docker安装

```bash
# 构建Docker镜像
docker build -t srt-adapter -f Dockerfile.srt .

# 运行容器
docker run -d --name srt-adapter \
  -e SRT_MODEL_PATH=/models \
  -e SRT_DATA_PATH=/data \
  -e SRT_USE_GPU=true \
  -v /path/to/local/models:/models \
  -v /path/to/local/data:/data \
  --gpus all \
  -p 8082:8082 \
  srt-adapter
```

## 5. 配置文件示例

```yaml
# config/srt.yaml
model:
  path: ${SRT_MODEL_PATH}
  type: "transformer"
  hidden_size: 768
  num_layers: 12
  num_heads: 12

data:
  path: ${SRT_DATA_PATH}
  validation_split: 0.1
  test_split: 0.1

training:
  batch_size: ${SRT_BATCH_SIZE:32}
  learning_rate: ${SRT_LEARNING_RATE:0.001}
  max_iterations: ${SRT_MAX_ITERATIONS:1000}
  optimizer: "adam"
  scheduler: "cosine"
  gradient_clip: 1.0

resources:
  num_workers: ${SRT_NUM_WORKERS:4}
  use_gpu: ${SRT_USE_GPU:true}
  gpu_device: ${SRT_GPU_DEVICE:0}
  mixed_precision: true

logging:
  level: ${SRT_LOG_LEVEL:INFO}
  file: ${SRT_LOG_FILE:/var/log/srt-adapter.log}
  tensorboard: true
  tensorboard_dir: ${SRT_TENSORBOARD_DIR:/tmp/srt_tensorboard}

features:
  self_reward: true
  rloo_optimization: true
  thought_improvement: true
```

## 6. 验证安装

```bash
# 验证SRT模型加载
python -m scripts.verify_srt_model

# 验证GPU支持
python -m scripts.verify_gpu_support

# 运行简单训练测试
python -m scripts.test_srt_training
```

## 7. 故障排除

### 7.1 常见问题

- **CUDA错误**: 检查CUDA版本与PyTorch版本是否匹配
- **内存不足**: 减小批处理大小或使用梯度累积
- **模型加载失败**: 检查模型路径和权限
- **训练速度慢**: 启用混合精度训练或增加批处理大小

### 7.2 日志位置

- 应用日志: `/var/log/srt-adapter.log`
- TensorBoard日志: `/tmp/srt_tensorboard`
- Docker日志: `docker logs srt-adapter`

## 8. 更新与维护

```bash
# 更新依赖
pip install -U srt-learning rloo-algorithm

# 备份模型
cp -r ${SRT_MODEL_PATH} ${SRT_MODEL_PATH}_backup_$(date +%Y%m%d)

# 备份配置
cp config/srt.yaml config/srt.yaml.bak
```

## 9. 预训练模型

SRT提供以下预训练模型，可以直接使用：

- `srt-base`: 基础模型，适用于一般场景
- `srt-large`: 大型模型，提供更高精度
- `srt-domain-specific`: 领域特定模型，针对特定任务优化

下载预训练模型：

```bash
# 下载基础模型
python -m scripts.download_model --model srt-base --output ${SRT_MODEL_PATH}

# 下载大型模型
python -m scripts.download_model --model srt-large --output ${SRT_MODEL_PATH}
```
