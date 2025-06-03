# PowerAutomation 与 Supermemory API 集成方案

## 1. 概述

本文档详细说明了 PowerAutomation 与 supermemory.ai 无限记忆 API 的集成方案，包括架构设计、接口定义、实现细节和使用示例。通过集成 supermemory.ai API，PowerAutomation 系统获得了强大的无限上下文记忆能力，能够更好地支持通用智能体的六大特性和五大治理原则。

## 2. 集成架构

PowerAutomation 系统通过以下组件与 supermemory.ai 进行集成：

1. **无限上下文适配器**：`mcptool/adapters/infinite_context_adapter.py`
2. **记忆管理服务**：`backend/services/memory_service.py`
3. **前端记忆组件**：`frontend/src/utils/memory-manager.js`

这些组件协同工作，确保用户查询和系统思考过程能够被正确存储和检索，实现无限上下文记忆功能。

### 2.1 三层架构中的位置

在 PowerAutomation 的三层架构中，supermemory 集成位于应用层，与其他应用层组件（如 Agent Problem Solver、Thought Action Recorder 等）协同工作，并通过增强层的 Evolution Algorithms 与基础层的组件进行交互。

![PowerAutomation 三层架构](/home/ubuntu/powerautomation_integration/phase1/images/powerautomation_architecture.png)

### 2.2 与六大特性的关系

supermemory API 集成支持通用智能体的六大特性：

1. **平台特性**：提供跨平台的记忆存储和检索能力
2. **UI布局特性**：支持记忆数据的可视化展示
3. **提示词特性**：增强提示词的上下文理解
4. **思维特性**：记录和优化思考过程
5. **内容特性**：提供丰富的内容生成支持
6. **记忆特性**：实现无限上下文记忆，是核心支持点

### 2.3 与五大治理原则的关系

supermemory API 集成遵循五大治理原则：

1. **结构保护原则**：确保记忆数据的结构完整性
2. **兼容性原则**：支持多种数据格式和接口标准
3. **空间利用原则**：优化记忆存储和检索效率
4. **模块化原则**：采用适配器模式，便于扩展和替换
5. **一致性原则**：确保记忆数据的一致性和可靠性

## 3. API Key 配置

### 3.1 获取 API Key

1. 访问 [supermemory.ai](https://supermemory.ai) 官方网站
2. 注册或登录您的账户
3. 导航至 API 管理页面
4. 创建新的 API Key，选择适当的权限级别
5. 复制生成的 API Key

### 3.2 配置 API Key

#### 3.2.1 环境变量配置（推荐）

在系统环境中设置以下环境变量：

```bash
# Linux/macOS
export SUPERMEMORY_API_KEY="sm_ohYKVYxdyurx5qGri5VqCi_pHvPpGCBJXgePHmQffwwICeZhiFcCZlKSrQLLPcZAqRaIpjGyiQnIIiTEbPkpWuH"
export SUPERMEMORY_API_URL="https://api.supermemory.ai/v1"

# Windows
set SUPERMEMORY_API_KEY=sm_ohYKVYxdyurx5qGri5VqCi_pHvPpGCBJXgePHmQffwwICeZhiFcCZlKSrQLLPcZAqRaIpjGyiQnIIiTEbPkpWuH
set SUPERMEMORY_API_URL=https://api.supermemory.ai/v1
```

对于生产环境，建议将这些环境变量添加到系统的启动脚本或服务配置中。

#### 3.2.2 配置文件配置

或者，您可以在 `config/api_keys.json` 文件中配置 API Key：

```json
{
  "supermemory": {
    "api_key": "sm_ohYKVYxdyurx5qGri5VqCi_pHvPpGCBJXgePHmQffwwICeZhiFcCZlKSrQLLPcZAqRaIpjGyiQnIIiTEbPkpWuH",
    "api_url": "https://api.supermemory.ai/v1"
  }
}
```

**注意**：确保 `api_keys.json` 文件不被提交到版本控制系统中，以保护 API Key 的安全。

### 3.3 API Key 轮换

为了安全起见，建议定期轮换 API Key：

1. 在 supermemory.ai 管理页面创建新的 API Key
2. 更新系统中的 API Key 配置
3. 验证系统功能正常
4. 在 supermemory.ai 管理页面删除旧的 API Key

## 4. 无限上下文适配器配置

无限上下文适配器 (`infinite_context_adapter.py`) 是系统与 supermemory.ai API 交互的核心组件。以下是其主要配置选项：

```python
# mcptool/adapters/infinite_context_adapter.py

class InfiniteContextAdapter:
    def __init__(self, config=None):
        self.config = config or {}
        self.api_key = self.config.get('api_key') or os.environ.get('SUPERMEMORY_API_KEY')
        self.api_url = self.config.get('api_url') or os.environ.get('SUPERMEMORY_API_URL', 'https://api.supermemory.ai/v1')
        self.max_tokens = self.config.get('max_tokens', 100000)
        self.compression_ratio = self.config.get('compression_ratio', 0.5)
        
        if not self.api_key:
            raise ValueError("Supermemory API Key not found. Please set SUPERMEMORY_API_KEY environment variable or provide in config.")
```

### 4.1 基本配置

您可以通过修改 `config/memory_config.json` 文件来调整无限上下文适配器的高级配置：

```json
{
  "memory": {
    "max_tokens": 100000,
    "compression_ratio": 0.5,
    "cache_ttl": 3600,
    "priority_weights": {
      "recency": 0.7,
      "relevance": 0.3
    },
    "storage_strategy": "hybrid",
    "indexing_method": "semantic"
  }
}
```

### 4.2 高级配置

无限上下文适配器支持以下高级配置选项：

- **max_tokens**：记忆存储的最大令牌数
- **compression_ratio**：记忆压缩比例
- **cache_ttl**：缓存生存时间（秒）
- **priority_weights**：记忆优先级权重
  - **recency**：基于时间的权重
  - **relevance**：基于相关性的权重
- **storage_strategy**：存储策略（hybrid、semantic、keyword）
- **indexing_method**：索引方法（semantic、keyword、hybrid）

## 5. API 使用示例

### 5.1 存储记忆

以下是使用无限上下文适配器存储记忆的示例：

```python
from mcptool.adapters.infinite_context_adapter import InfiniteContextAdapter

# 创建适配器实例
adapter = InfiniteContextAdapter()

# 存储记忆
memory_id = adapter.store_memory({
    "query": "如何优化PowerAutomation的UI布局特性？",
    "response": "PowerAutomation的UI布局可以通过以下方式优化...",
    "features": {
        "platform_feature": "PowerAutomation自动化平台",
        "ui_layout": "两栏布局，左侧为导航栏，右侧为主内容区",
        "prompt": "用户输入优化UI布局的请求",
        "thinking": "分析用户需求，确定UI优化方向",
        "content": "生成UI优化建议",
        "memory": "记录用户查询和系统思考过程"
    }
})

print(f"Memory stored with ID: {memory_id}")
```

### 5.2 检索记忆

以下是使用无限上下文适配器检索记忆的示例：

```python
# 检索记忆
memories = adapter.retrieve_memories("UI布局优化", limit=5)

for memory in memories:
    print(f"Query: {memory['query']}")
    print(f"Response: {memory['response']}")
    print(f"Features: {memory['features']}")
    print("---")
```

### 5.3 前端集成

以下是前端记忆管理组件的示例：

```javascript
// frontend/src/utils/memory-manager.js

class MemoryManager {
  constructor() {
    this.apiUrl = process.env.REACT_APP_SUPERMEMORY_API_URL || 'https://api.supermemory.ai/v1';
    this.headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.REACT_APP_SUPERMEMORY_API_KEY}`
    };
  }

  async storeMemory(data) {
    try {
      const response = await fetch(`${this.apiUrl}/memories`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(data)
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error storing memory:', error);
      throw error;
    }
  }

  async retrieveMemories(query, limit = 5) {
    try {
      const response = await fetch(`${this.apiUrl}/memories/search?q=${encodeURIComponent(query)}&limit=${limit}`, {
        method: 'GET',
        headers: this.headers
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error retrieving memories:', error);
      throw error;
    }
  }
}

export default new MemoryManager();
```

## 6. 与自动化测试工作流集成

自动化测试工作流可以利用 supermemory API 记录测试过程和结果，提高测试效率和质量。

### 6.1 测试记忆存储

```python
def store_test_memory(test_name, test_input, test_output, test_result):
    """存储测试记忆"""
    adapter = InfiniteContextAdapter()
    
    memory_id = adapter.store_memory({
        "query": f"测试用例: {test_name}",
        "response": f"测试结果: {test_result}",
        "features": {
            "platform_feature": "PowerAutomation自动化测试平台",
            "ui_layout": "测试控制台界面",
            "prompt": f"测试输入: {test_input}",
            "thinking": "测试执行过程",
            "content": f"测试输出: {test_output}",
            "memory": "记录测试过程和结果"
        }
    })
    
    return memory_id
```

### 6.2 测试记忆检索

```python
def retrieve_similar_tests(test_name, limit=5):
    """检索相似测试记忆"""
    adapter = InfiniteContextAdapter()
    
    memories = adapter.retrieve_memories(f"测试用例: {test_name}", limit=limit)
    
    return memories
```

## 7. 与自动化智能体设计工作流集成

自动化智能体设计工作流可以利用 supermemory API 记录设计过程和决策，提高智能体设计的效率和质量。

### 7.1 智能体设计记忆存储

```python
def store_agent_design_memory(agent_name, design_input, design_output):
    """存储智能体设计记忆"""
    adapter = InfiniteContextAdapter()
    
    memory_id = adapter.store_memory({
        "query": f"智能体设计: {agent_name}",
        "response": f"设计输出: {design_output}",
        "features": {
            "platform_feature": "PowerAutomation智能体设计平台",
            "ui_layout": "设计工作台界面",
            "prompt": f"设计输入: {design_input}",
            "thinking": "设计思考过程",
            "content": "设计方案和实现代码",
            "memory": "记录设计过程和决策"
        }
    })
    
    return memory_id
```

### 7.2 智能体设计记忆检索

```python
def retrieve_similar_designs(agent_name, limit=5):
    """检索相似智能体设计记忆"""
    adapter = InfiniteContextAdapter()
    
    memories = adapter.retrieve_memories(f"智能体设计: {agent_name}", limit=limit)
    
    return memories
```

## 8. 与多模型协同层集成

多模型协同层可以利用 supermemory API 记录模型协同过程和结果，提高模型协同的效率和质量。

### 8.1 模型协同记忆存储

```python
def store_model_synergy_memory(task_name, models_used, synergy_result):
    """存储模型协同记忆"""
    adapter = InfiniteContextAdapter()
    
    memory_id = adapter.store_memory({
        "query": f"模型协同任务: {task_name}",
        "response": f"协同结果: {synergy_result}",
        "features": {
            "platform_feature": "PowerAutomation多模型协同平台",
            "ui_layout": "协同控制台界面",
            "prompt": f"使用模型: {models_used}",
            "thinking": "模型协同过程",
            "content": "协同输出内容",
            "memory": "记录协同过程和结果"
        }
    })
    
    return memory_id
```

### 8.2 模型协同记忆检索

```python
def retrieve_similar_synergies(task_name, limit=5):
    """检索相似模型协同记忆"""
    adapter = InfiniteContextAdapter()
    
    memories = adapter.retrieve_memories(f"模型协同任务: {task_name}", limit=limit)
    
    return memories
```

## 9. 治理原则实现

supermemory API 集成严格遵循五大治理原则，以下是具体实现：

### 9.1 结构保护原则

```python
def validate_memory_structure(memory_data):
    """验证记忆数据结构"""
    required_fields = ["query", "response", "features"]
    for field in required_fields:
        if field not in memory_data:
            raise ValueError(f"Missing required field: {field}")
    
    required_features = ["platform_feature", "ui_layout", "prompt", "thinking", "content", "memory"]
    for feature in required_features:
        if feature not in memory_data["features"]:
            raise ValueError(f"Missing required feature: {feature}")
    
    return True
```

### 9.2 兼容性原则

```python
def convert_memory_format(memory_data, target_format="standard"):
    """转换记忆数据格式"""
    if target_format == "standard":
        return memory_data
    elif target_format == "simple":
        return {
            "query": memory_data["query"],
            "response": memory_data["response"]
        }
    elif target_format == "detailed":
        return {
            "query": memory_data["query"],
            "response": memory_data["response"],
            "features": memory_data["features"],
            "metadata": {
                "timestamp": memory_data.get("timestamp", time.time()),
                "source": memory_data.get("source", "unknown")
            }
        }
    else:
        raise ValueError(f"Unsupported target format: {target_format}")
```

### 9.3 空间利用原则

```python
def optimize_memory_storage(memory_data, compression_level="medium"):
    """优化记忆存储空间"""
    if compression_level == "low":
        # 简单压缩，只保留必要字段
        return {
            "query": memory_data["query"],
            "response": memory_data["response"],
            "features": memory_data["features"]
        }
    elif compression_level == "medium":
        # 中等压缩，压缩长文本
        return {
            "query": memory_data["query"],
            "response": _compress_text(memory_data["response"]),
            "features": memory_data["features"]
        }
    elif compression_level == "high":
        # 高度压缩，压缩所有文本并移除非必要字段
        return {
            "query": _compress_text(memory_data["query"]),
            "response": _compress_text(memory_data["response"]),
            "features": {
                k: _compress_text(v) if isinstance(v, str) else v
                for k, v in memory_data["features"].items()
                if k in ["platform_feature", "memory"]
            }
        }
    else:
        raise ValueError(f"Unsupported compression level: {compression_level}")

def _compress_text(text, ratio=0.5):
    """压缩文本"""
    if len(text) <= 100:
        return text
    
    # 简单实现，实际应使用更复杂的算法
    words = text.split()
    compressed_length = max(10, int(len(words) * ratio))
    return " ".join(words[:compressed_length]) + "..."
```

### 9.4 模块化原则

```python
class MemoryAdapterFactory:
    """记忆适配器工厂"""
    @staticmethod
    def create_adapter(adapter_type="infinite_context"):
        """创建记忆适配器"""
        if adapter_type == "infinite_context":
            from mcptool.adapters.infinite_context_adapter import InfiniteContextAdapter
            return InfiniteContextAdapter()
        elif adapter_type == "local_memory":
            from mcptool.adapters.local_memory_adapter import LocalMemoryAdapter
            return LocalMemoryAdapter()
        elif adapter_type == "redis_memory":
            from mcptool.adapters.redis_memory_adapter import RedisMemoryAdapter
            return RedisMemoryAdapter()
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
```

### 9.5 一致性原则

```python
class MemoryConsistencyManager:
    """记忆一致性管理器"""
    def __init__(self, adapter):
        self.adapter = adapter
        self.cache = {}
    
    def store_memory(self, memory_data):
        """存储记忆并确保一致性"""
        # 验证数据结构
        validate_memory_structure(memory_data)
        
        # 检查重复
        existing_memories = self.adapter.retrieve_memories(memory_data["query"], limit=1)
        if existing_memories and self._is_duplicate(memory_data, existing_memories[0]):
            return existing_memories[0]["id"]
        
        # 存储记忆
        memory_id = self.adapter.store_memory(memory_data)
        
        # 更新缓存
        self.cache[memory_id] = memory_data
        
        return memory_id
    
    def _is_duplicate(self, memory1, memory2, similarity_threshold=0.9):
        """检查两个记忆是否重复"""
        # 简单实现，实际应使用更复杂的相似度算法
        query_similarity = self._calculate_similarity(memory1["query"], memory2["query"])
        response_similarity = self._calculate_similarity(memory1["response"], memory2["response"])
        
        return query_similarity > similarity_threshold and response_similarity > similarity_threshold
    
    def _calculate_similarity(self, text1, text2):
        """计算两个文本的相似度"""
        # 简单实现，实际应使用更复杂的相似度算法
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
```

## 10. 总结

PowerAutomation 与 supermemory.ai 无限记忆 API 的集成为系统提供了强大的无限上下文记忆能力，支持通用智能体的六大特性和五大治理原则。通过无限上下文适配器、记忆管理服务和前端记忆组件，系统能够高效地存储和检索记忆数据，提高智能体的性能和用户体验。

集成方案采用适配器模式，最小化对原有代码的修改，实现了系统的灵活性和可扩展性。通过定义标准接口和实现适配器，使系统能够灵活地使用不同的底层组件，提高了系统的可维护性。

后续工作将重点完善记忆数据的压缩和优化算法，提高记忆存储和检索的效率，进一步提升系统的性能和用户体验。
