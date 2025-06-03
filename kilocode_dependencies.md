# Kilo Code 依赖配置

## 1. 依赖包

```
kilo-code==1.2.3
mcp-server-connector==0.9.1
requests>=2.25.0
pyyaml>=5.4.1
```

## 2. 环境变量配置

```bash
# Kilo Code API配置
KILO_CODE_API_KEY=your_api_key_here
KILO_CODE_SERVER_URL=https://api.kilocode.ai/v1
KILO_CODE_TIMEOUT=30

# MCP Server配置
MCP_SERVER_URL=http://localhost:8080
MCP_SERVER_API_KEY=your_mcp_api_key_here

# 日志配置
KILO_CODE_LOG_LEVEL=INFO
KILO_CODE_LOG_FILE=/var/log/kilocode-adapter.log
```

## 3. 资源需求

- **CPU**: 最低2核，推荐4核
- **内存**: 最低4GB，推荐8GB
- **存储**: 最低10GB可用空间
- **网络**: 稳定的互联网连接，带宽至少10Mbps

## 4. 安装步骤

### 4.1 使用pip安装

```bash
# 创建虚拟环境
python -m venv kilocode-env
source kilocode-env/bin/activate

# 安装依赖
pip install -r requirements-kilocode.txt
```

### 4.2 使用Docker安装

```bash
# 构建Docker镜像
docker build -t kilocode-adapter -f Dockerfile.kilocode .

# 运行容器
docker run -d --name kilocode-adapter \
  -e KILO_CODE_API_KEY=your_api_key_here \
  -e KILO_CODE_SERVER_URL=https://api.kilocode.ai/v1 \
  -e MCP_SERVER_URL=http://localhost:8080 \
  -p 8081:8081 \
  kilocode-adapter
```

## 5. 配置文件示例

```yaml
# config/kilocode.yaml
api:
  key: ${KILO_CODE_API_KEY}
  server_url: ${KILO_CODE_SERVER_URL}
  timeout: ${KILO_CODE_TIMEOUT:30}

mcp:
  server_url: ${MCP_SERVER_URL}
  api_key: ${MCP_SERVER_API_KEY}

logging:
  level: ${KILO_CODE_LOG_LEVEL:INFO}
  file: ${KILO_CODE_LOG_FILE:/var/log/kilocode-adapter.log}

features:
  code_generation: true
  code_interpretation: true
  task_decomposition: true

performance:
  cache_size: 1000
  max_tokens: 8192
  batch_size: 10
```

## 6. 验证安装

```bash
# 验证Kilo Code API连接
python -m scripts.verify_kilocode_connection

# 验证MCP Server连接
python -m scripts.verify_mcp_connection
```

## 7. 故障排除

### 7.1 常见问题

- **API连接失败**: 检查API密钥和服务器URL是否正确
- **超时错误**: 增加超时设置或检查网络连接
- **内存不足**: 增加系统内存或减小批处理大小

### 7.2 日志位置

- 应用日志: `/var/log/kilocode-adapter.log`
- Docker日志: `docker logs kilocode-adapter`

## 8. 更新与维护

```bash
# 更新依赖
pip install -U kilo-code mcp-server-connector

# 备份配置
cp config/kilocode.yaml config/kilocode.yaml.bak
```
