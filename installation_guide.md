# PowerAutomation 集成方案安装指南

本文档提供了 PowerAutomation 与 Kilo Code、SRT 集成方案的详细安装和配置指南。

## 1. 环境要求

### 1.1 基础环境

- Python 3.8+ (推荐 Python 3.11)
- pip 21.0+
- 操作系统：Linux (推荐 Ubuntu 20.04+)、macOS 或 Windows 10+

### 1.2 硬件要求

- **CPU 版本**：
  - 至少 4GB RAM
  - 4+ CPU 核心
  - 10GB 可用磁盘空间

- **GPU 版本**（可选，用于加速 SRT 训练）：
  - NVIDIA GPU，至少 4GB 显存
  - CUDA 11.7+ 和 cuDNN 8.5+

## 2. 安装步骤

### 2.1 克隆代码仓库

```bash
git clone https://github.com/your-organization/powerautomation.git
cd powerautomation
```

### 2.2 创建并激活虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Linux/macOS）
source venv/bin/activate

# 激活虚拟环境（Windows）
venv\Scripts\activate
```

### 2.3 安装依赖

#### 2.3.1 基础依赖

```bash
# 安装基础依赖
pip install -r requirements.txt
```

#### 2.3.2 PyTorch 安装

**CPU 版本**：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本**（NVIDIA GPU）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.4 安装集成组件

```bash
# 安装集成组件
cd powerautomation_integration
pip install -e .
```

## 3. 配置

### 3.1 配置文件

集成方案使用 JSON 格式的配置文件，默认路径为 `config/integration_config.json`。

#### 3.1.1 基础配置示例

```json
{
  "kilocode": {
    "api_key": "your_kilocode_api_key",
    "api_url": "https://api.kilocode.example.com/v1",
    "timeout": 30,
    "retry_attempts": 3
  },
  "srt": {
    "model_path": "models/srt_model.pt",
    "batch_size": 32,
    "learning_rate": 1e-5,
    "device": "cpu"
  },
  "logging": {
    "level": "INFO",
    "file": "logs/integration.log"
  }
}
```

#### 3.1.2 高级配置示例

```json
{
  "kilocode": {
    "api_key": "your_kilocode_api_key",
    "api_url": "https://api.kilocode.example.com/v1",
    "timeout": 30,
    "retry_attempts": 3,
    "max_tokens": 4096,
    "temperature": 0.7,
    "cache_dir": "cache/kilocode",
    "cache_ttl": 3600
  },
  "srt": {
    "model_path": "models/srt_model.pt",
    "batch_size": 32,
    "learning_rate": 1e-5,
    "device": "cuda",
    "max_iterations": 1000,
    "evaluation_interval": 100,
    "save_interval": 500,
    "optimizer": {
      "type": "Adam",
      "weight_decay": 0.01,
      "beta1": 0.9,
      "beta2": 0.999
    },
    "model_config": {
      "hidden_size": 768,
      "num_layers": 12,
      "dropout": 0.1
    }
  },
  "logging": {
    "level": "INFO",
    "file": "logs/integration.log",
    "rotation": "daily",
    "retention": 7,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  },
  "integration": {
    "mcp_coordinator_url": "http://localhost:8000",
    "rl_factory_path": "path/to/rl_factory",
    "max_concurrent_requests": 10,
    "request_timeout": 60
  }
}
```

### 3.2 环境变量

以下环境变量可用于覆盖配置文件中的设置：

```bash
# Kilo Code 配置
export KILOCODE_API_KEY="your_kilocode_api_key"
export KILOCODE_API_URL="https://api.kilocode.example.com/v1"

# SRT 配置
export SRT_MODEL_PATH="models/srt_model.pt"
export SRT_USE_CUDA="1"  # 使用 GPU，设为 "0" 则使用 CPU

# 日志配置
export INTEGRATION_LOG_LEVEL="INFO"
export INTEGRATION_LOG_FILE="logs/integration.log"
```

## 4. 验证安装

### 4.1 运行健康检查

```bash
python cli_testing/test_adapter.py --adapter all --command health_check
```

### 4.2 运行基础功能测试

```bash
python cli_testing/real_srt_test.py --test
```

### 4.3 运行集成测试

```bash
python cli_testing/real_srt_test.py --integration
```

### 4.4 运行扩展测试

```bash
python cli_testing/extended_test.py --all
```

## 5. 常见问题

### 5.1 PyTorch 安装问题

**问题**：安装 PyTorch 时出现内存不足错误。

**解决方案**：
- 尝试使用更小的 PyTorch 版本：`pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
- 增加系统交换空间
- 使用具有更多内存的机器

### 5.2 导入错误

**问题**：运行测试时出现 `ImportError: No module named 'adapters'`。

**解决方案**：
- 确保当前目录是项目根目录
- 检查是否已安装集成组件：`pip install -e .`
- 检查 `__init__.py` 文件是否存在于所有包目录中

### 5.3 API 连接问题

**问题**：无法连接到 Kilo Code API。

**解决方案**：
- 检查 API 密钥和 URL 是否正确
- 检查网络连接
- 检查防火墙设置
- 增加超时时间：在配置文件中设置 `"timeout": 60`

### 5.4 GPU 相关问题

**问题**：无法使用 GPU 加速。

**解决方案**：
- 确认 CUDA 和 cuDNN 已正确安装
- 检查 GPU 驱动版本是否与 CUDA 版本兼容
- 使用 `nvidia-smi` 命令检查 GPU 状态
- 确保环境变量 `SRT_USE_CUDA="1"` 已设置

## 6. 更新与升级

### 6.1 更新代码

```bash
git pull origin main
```

### 6.2 更新依赖

```bash
pip install -r requirements.txt --upgrade
```

### 6.3 更新集成组件

```bash
cd powerautomation_integration
pip install -e . --upgrade
```

## 7. 卸载

```bash
# 卸载集成组件
pip uninstall powerautomation-integration

# 删除代码仓库
cd ..
rm -rf powerautomation
```
