# 适配器层设计文档

## 1. 设计原则

为了最小化对原有代码库的改动，我们采用适配器层（Adapter Layer）模式进行Kilo Code和SRT的集成。主要设计原则如下：

1. **隔离变化**：通过适配器隔离外部组件的变化，保护核心代码不受影响
2. **接口标准化**：定义清晰的接口标准，确保组件间通信一致性
3. **最小侵入性**：避免直接修改原有代码，通过适配器桥接新旧系统
4. **可测试性**：适配器设计应便于单元测试和集成测试
5. **可扩展性**：支持未来可能的组件替换或升级

## 2. 适配器层架构

```
┌─────────────────────────┐      ┌─────────────────────────┐
│                         │      │                         │
│   PowerAutomation       │      │   External Components   │
│   Original Codebase     │      │                         │
│                         │      │  ┌─────────────────┐    │
│  ┌─────────────────┐    │      │  │                 │    │
│  │                 │    │      │  │   Kilo Code     │    │
│  │   MCP Tool      │    │      │  │                 │    │
│  │                 │    │      │  └─────────────────┘    │
│  └────────┬────────┘    │      │                         │
│           │             │      │  ┌─────────────────┐    │
│  ┌────────┴────────┐    │      │  │                 │    │
│  │                 │    │      │  │      SRT        │    │
│  │   RL Factory    │    │      │  │                 │    │
│  │                 │    │      │  └─────────────────┘    │
│  └─────────────────┘    │      │                         │
│                         │      └─────────────────────────┘
└──────────┬──────────────┘                 ▲
           │                                 │
           │                                 │
           ▼                                 │
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                    Adapter Layer                        │
│                                                         │
│  ┌─────────────────┐         ┌─────────────────────┐    │
│  │                 │         │                     │    │
│  │ KiloCodeAdapter │         │   SRTAdapter        │    │
│  │                 │         │                     │    │
│  └─────────────────┘         └─────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 3. 接口定义

### 3.1 Kilo Code 适配器接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class CodeGenerationInterface(ABC):
    """代码生成接口标准"""
    
    @abstractmethod
    def generate_code(self, prompt: str, context: Optional[Dict[str, Any]] = None, 
                     mode: str = "standard") -> str:
        """
        根据提示生成代码
        
        Args:
            prompt: 代码生成提示
            context: 上下文信息
            mode: 生成模式，可选值包括 "standard", "optimized", "explained"
            
        Returns:
            生成的代码字符串
        """
        pass
    
    @abstractmethod
    def interpret_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        解释代码
        
        Args:
            code: 需要解释的代码
            context: 上下文信息
            
        Returns:
            包含代码解释的字典
        """
        pass
    
    @abstractmethod
    def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        分解任务
        
        Args:
            task_description: 任务描述
            
        Returns:
            分解后的子任务列表
        """
        pass
```

### 3.2 SRT 适配器接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class SelfRewardTrainingInterface(ABC):
    """自我奖励训练接口标准"""
    
    @abstractmethod
    def train(self, thought_process: Union[str, Dict[str, Any]], 
             iterations: int = 100) -> Dict[str, Any]:
        """
        使用自我奖励机制训练模型
        
        Args:
            thought_process: 思考过程，可以是字符串或结构化数据
            iterations: 训练迭代次数
            
        Returns:
            训练结果信息
        """
        pass
    
    @abstractmethod
    def evaluate(self, thought_process: Union[str, Dict[str, Any]]) -> float:
        """
        评估思考过程的质量
        
        Args:
            thought_process: 需要评估的思考过程
            
        Returns:
            质量评分（0-1之间的浮点数）
        """
        pass
    
    @abstractmethod
    def improve(self, thought_process: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        改进思考过程
        
        Args:
            thought_process: 原始思考过程
            
        Returns:
            改进后的思考过程
        """
        pass
```

## 4. 适配器实现策略

### 4.1 Kilo Code 适配器

```python
from interfaces import CodeGenerationInterface
from typing import List, Dict, Any, Optional
import kilo_code  # 假设的Kilo Code库

class KiloCodeAdapter(CodeGenerationInterface):
    """Kilo Code适配器实现"""
    
    def __init__(self, api_key: Optional[str] = None, server_url: Optional[str] = None):
        """
        初始化Kilo Code适配器
        
        Args:
            api_key: Kilo Code API密钥
            server_url: Kilo Code服务器URL
        """
        self.client = kilo_code.KiloCodeClient(api_key=api_key, server_url=server_url)
    
    def generate_code(self, prompt: str, context: Optional[Dict[str, Any]] = None, 
                     mode: str = "standard") -> str:
        """实现代码生成接口"""
        return self.client.generate(prompt, context=context, mode=mode)
    
    def interpret_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """实现代码解释接口"""
        return self.client.interpret(code, context=context)
    
    def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """实现任务分解接口"""
        return self.client.decompose_task(task_description)
```

### 4.2 SRT 适配器

```python
from interfaces import SelfRewardTrainingInterface
from typing import List, Dict, Any, Optional, Union
import srt  # 假设的SRT库

class SRTAdapter(SelfRewardTrainingInterface):
    """SRT适配器实现"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化SRT适配器
        
        Args:
            model_path: SRT模型路径
        """
        self.model = srt.SRTModel.load(model_path) if model_path else srt.SRTModel()
        self.optimizer = srt.RLOOOptimizer()
    
    def train(self, thought_process: Union[str, Dict[str, Any]], 
             iterations: int = 100) -> Dict[str, Any]:
        """实现自我奖励训练接口"""
        results = {"iterations": iterations, "improvements": []}
        
        for i in range(iterations):
            reward = self.model.compute_self_reward(thought_process)
            self.model = self.optimizer.step(self.model, thought_process, reward)
            
            if i % 10 == 0:  # 每10次迭代记录一次改进
                results["improvements"].append({
                    "iteration": i,
                    "reward": reward,
                    "model_state": self.model.get_state_summary()
                })
        
        return results
    
    def evaluate(self, thought_process: Union[str, Dict[str, Any]]) -> float:
        """实现思考过程评估接口"""
        return self.model.evaluate(thought_process)
    
    def improve(self, thought_process: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """实现思考过程改进接口"""
        return self.model.improve(thought_process)
```

## 5. 集成点

### 5.1 MCP Tool 集成点

MCP Tool与Kilo Code的集成将通过以下方式实现：

1. 在MCP Tool的适配器目录下添加KiloCodeAdapter
2. 不修改MCP Tool核心代码，而是通过配置文件指定使用KiloCodeAdapter
3. 如果必须修改，仅在代码处理器中添加对适配器的调用，保持原有逻辑不变

```python
# 示例：在配置文件中指定使用KiloCodeAdapter
{
    "code_processor": {
        "adapter": "adapters.kilo_code_adapter.KiloCodeAdapter",
        "config": {
            "api_key": "${KILO_CODE_API_KEY}",
            "server_url": "${KILO_CODE_SERVER_URL}"
        }
    }
}
```

### 5.2 RL Factory 集成点

RL Factory与SRT的集成将通过以下方式实现：

1. 在RL Factory的适配器目录下添加SRTAdapter
2. 通过配置文件指定使用SRTAdapter进行训练和评估
3. 如果必须修改，仅在学习器中添加对适配器的调用，保持原有学习逻辑不变

```python
# 示例：在配置文件中指定使用SRTAdapter
{
    "learner": {
        "adapter": "adapters.srt_adapter.SRTAdapter",
        "config": {
            "model_path": "${SRT_MODEL_PATH}"
        }
    }
}
```

## 6. 必要的最小化改动

对于不得不改动的部分，我们将严格限制在以下范围：

1. **配置加载**：修改配置加载逻辑，支持适配器配置
2. **适配器初始化**：在启动时初始化适配器
3. **接口调用**：在必要的地方添加对适配器接口的调用

所有改动将遵循以下原则：
- 每处改动不超过5行代码
- 使用条件判断确保向后兼容
- 添加详细注释说明改动目的
- 提供回滚机制

## 7. 测试策略

为确保适配器层正常工作且不影响原有功能，我们将采用以下测试策略：

1. **单元测试**：测试每个适配器的独立功能
2. **集成测试**：测试适配器与原系统的集成
3. **回归测试**：确保原有功能不受影响
4. **性能测试**：评估适配器对系统性能的影响

## 8. 部署考虑

适配器层的部署将考虑以下因素：

1. **依赖管理**：明确声明Kilo Code和SRT的依赖
2. **环境变量**：使用环境变量配置适配器
3. **版本兼容性**：确保与原系统版本兼容
4. **回滚机制**：提供简单的回滚机制

## 9. 文档和维护

为确保适配器层的可维护性，我们将提供：

1. **详细的接口文档**：描述每个接口的用途和使用方法
2. **示例代码**：提供适配器使用的示例
3. **配置指南**：说明如何配置适配器
4. **故障排除指南**：常见问题及解决方法
