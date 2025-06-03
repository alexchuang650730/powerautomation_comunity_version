# 统一API接口标准文档

## 1. 概述

本文档定义了PowerAutomation集成Kilo Code和SRT的统一API接口标准。这些接口设计遵循适配器模式，旨在最小化对原有代码的修改，同时提供清晰、一致的集成点。

## 2. 接口设计原则

- **单一职责**: 每个接口专注于单一功能领域
- **接口隔离**: 客户端不应依赖它不使用的接口
- **依赖倒置**: 高层模块不应依赖低层模块，都应依赖于抽象
- **开闭原则**: 对扩展开放，对修改关闭
- **最小知识**: 接口应只暴露必要的方法和属性

## 3. 核心接口定义

### 3.1 代码生成与处理接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

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
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        批量生成代码
        
        Args:
            prompts: 代码生成提示列表
            context: 共享上下文信息
            
        Returns:
            生成的代码字符串列表
        """
        pass


class CodeOptimizationInterface(ABC):
    """代码优化接口标准"""
    
    @abstractmethod
    def optimize_code(self, code: str, optimization_level: str = "medium") -> str:
        """
        优化代码
        
        Args:
            code: 需要优化的代码
            optimization_level: 优化级别，可选值包括 "low", "medium", "high"
            
        Returns:
            优化后的代码
        """
        pass
    
    @abstractmethod
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """
        分析代码复杂度
        
        Args:
            code: 需要分析的代码
            
        Returns:
            包含复杂度分析的字典
        """
        pass
    
    @abstractmethod
    def suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """
        提供代码改进建议
        
        Args:
            code: 需要分析的代码
            
        Returns:
            改进建议列表
        """
        pass
```

### 3.2 自我奖励训练接口

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
    
    @abstractmethod
    def batch_train(self, thought_processes: List[Union[str, Dict[str, Any]]], 
                  batch_size: int = 32) -> Dict[str, Any]:
        """
        批量训练模型
        
        Args:
            thought_processes: 思考过程列表
            batch_size: 批处理大小
            
        Returns:
            训练结果信息
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """
        保存模型
        
        Args:
            path: 模型保存路径
            
        Returns:
            保存是否成功
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        加载模型
        
        Args:
            path: 模型加载路径
            
        Returns:
            加载是否成功
        """
        pass
```

### 3.3 强化学习优化接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable

class RLOptimizationInterface(ABC):
    """强化学习优化接口标准"""
    
    @abstractmethod
    def optimize(self, model: Any, environment: Any, 
               reward_function: Callable, iterations: int = 1000) -> Dict[str, Any]:
        """
        使用强化学习优化模型
        
        Args:
            model: 需要优化的模型
            environment: 训练环境
            reward_function: 奖励函数
            iterations: 优化迭代次数
            
        Returns:
            优化结果信息
        """
        pass
    
    @abstractmethod
    def evaluate_policy(self, model: Any, environment: Any, 
                      episodes: int = 10) -> Dict[str, Any]:
        """
        评估策略
        
        Args:
            model: 需要评估的模型
            environment: 评估环境
            episodes: 评估回合数
            
        Returns:
            评估结果信息
        """
        pass
    
    @abstractmethod
    def generate_trajectory(self, model: Any, environment: Any, 
                         max_steps: int = 1000) -> List[Dict[str, Any]]:
        """
        生成轨迹
        
        Args:
            model: 模型
            environment: 环境
            max_steps: 最大步数
            
        Returns:
            轨迹信息列表
        """
        pass
```

## 4. 适配器接口

### 4.1 Kilo Code适配器接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class KiloCodeAdapterInterface(ABC):
    """Kilo Code适配器接口标准"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化适配器
        
        Args:
            config: 配置信息
            
        Returns:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """
        获取能力信息
        
        Returns:
            支持的能力列表
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        关闭适配器
        
        Returns:
            关闭是否成功
        """
        pass
```

### 4.2 SRT适配器接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class SRTAdapterInterface(ABC):
    """SRT适配器接口标准"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化适配器
        
        Args:
            config: 配置信息
            
        Returns:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """
        获取能力信息
        
        Returns:
            支持的能力列表
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        关闭适配器
        
        Returns:
            关闭是否成功
        """
        pass
```

## 5. 事件与回调接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

class EventListenerInterface(ABC):
    """事件监听器接口标准"""
    
    @abstractmethod
    def on_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        事件处理
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        pass


class EventEmitterInterface(ABC):
    """事件发射器接口标准"""
    
    @abstractmethod
    def add_listener(self, event_type: str, listener: EventListenerInterface) -> bool:
        """
        添加事件监听器
        
        Args:
            event_type: 事件类型
            listener: 事件监听器
            
        Returns:
            添加是否成功
        """
        pass
    
    @abstractmethod
    def remove_listener(self, event_type: str, listener: EventListenerInterface) -> bool:
        """
        移除事件监听器
        
        Args:
            event_type: 事件类型
            listener: 事件监听器
            
        Returns:
            移除是否成功
        """
        pass
    
    @abstractmethod
    def emit(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        发射事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        pass
```

## 6. 配置与工厂接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, TypeVar

T = TypeVar('T')

class ConfigurationInterface(ABC):
    """配置接口标准"""
    
    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            path: 配置文件路径
            
        Returns:
            配置信息
        """
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], path: str) -> bool:
        """
        保存配置
        
        Args:
            config: 配置信息
            path: 配置文件路径
            
        Returns:
            保存是否成功
        """
        pass
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置项键
            default: 默认值
            
        Returns:
            配置项值
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置项键
            value: 配置项值
        """
        pass


class AdapterFactoryInterface(ABC):
    """适配器工厂接口标准"""
    
    @abstractmethod
    def create_adapter(self, adapter_type: str, config: Dict[str, Any]) -> Any:
        """
        创建适配器
        
        Args:
            adapter_type: 适配器类型
            config: 配置信息
            
        Returns:
            创建的适配器实例
        """
        pass
    
    @abstractmethod
    def register_adapter_class(self, adapter_type: str, adapter_class: Type[T]) -> bool:
        """
        注册适配器类
        
        Args:
            adapter_type: 适配器类型
            adapter_class: 适配器类
            
        Returns:
            注册是否成功
        """
        pass
    
    @abstractmethod
    def get_registered_adapters(self) -> Dict[str, Type[T]]:
        """
        获取已注册的适配器
        
        Returns:
            适配器类型到适配器类的映射
        """
        pass
```

## 7. 接口使用示例

### 7.1 使用Kilo Code适配器

```python
# 创建适配器实例
adapter_factory = AdapterFactory()
kilo_code_adapter = adapter_factory.create_adapter("kilo_code", config)

# 使用适配器生成代码
prompt = "Write a function to calculate fibonacci numbers"
code = kilo_code_adapter.generate_code(prompt)

# 解释代码
interpretation = kilo_code_adapter.interpret_code(code)

# 分解任务
task = "Build a web scraper for news articles"
subtasks = kilo_code_adapter.decompose_task(task)
```

### 7.2 使用SRT适配器

```python
# 创建适配器实例
adapter_factory = AdapterFactory()
srt_adapter = adapter_factory.create_adapter("srt", config)

# 训练模型
thought_process = "First, I need to understand the requirements..."
training_result = srt_adapter.train(thought_process, iterations=100)

# 评估思考过程
quality = srt_adapter.evaluate(thought_process)

# 改进思考过程
improved_thought = srt_adapter.improve(thought_process)
```

## 8. 接口版本控制

为确保接口的稳定性和向后兼容性，我们采用语义化版本控制：

- **主版本号**：不兼容的API变更
- **次版本号**：向后兼容的功能性新增
- **修订号**：向后兼容的问题修正

当前接口版本：**1.0.0**

## 9. 接口扩展指南

### 9.1 添加新方法

1. 在接口类中添加新的抽象方法
2. 为所有实现类添加该方法的实现
3. 更新版本号（次版本号+1）

### 9.2 添加新接口

1. 创建新的接口类
2. 实现该接口的适配器类
3. 在适配器工厂中注册新接口
4. 更新版本号（次版本号+1）

### 9.3 修改现有接口

1. 创建新版本的接口类
2. 实现新版本接口的适配器类
3. 保留旧版本接口的支持
4. 更新版本号（主版本号+1）

## 10. 接口测试指南

### 10.1 单元测试

```python
import unittest
from interfaces import CodeGenerationInterface
from adapters import KiloCodeAdapter

class TestKiloCodeAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = KiloCodeAdapter(api_key="test_key")
        
    def test_generate_code(self):
        prompt = "Write a function to calculate fibonacci numbers"
        code = self.adapter.generate_code(prompt)
        self.assertIsNotNone(code)
        self.assertIn("def fibonacci", code)
        
    def test_interpret_code(self):
        code = "def add(a, b): return a + b"
        interpretation = self.adapter.interpret_code(code)
        self.assertIsNotNone(interpretation)
        self.assertIn("adds two numbers", interpretation.lower())
```

### 10.2 模拟测试

```python
import unittest
from unittest.mock import Mock
from interfaces import CodeGenerationInterface

class TestWithMock(unittest.TestCase):
    def setUp(self):
        self.mock_adapter = Mock(spec=CodeGenerationInterface)
        self.mock_adapter.generate_code.return_value = "def fibonacci(n): pass"
        
    def test_with_mock(self):
        code = self.mock_adapter.generate_code("Write fibonacci")
        self.assertEqual(code, "def fibonacci(n): pass")
        self.mock_adapter.generate_code.assert_called_once()
```

## 11. 接口文档生成

使用以下命令生成接口文档：

```bash
# 使用Sphinx生成文档
sphinx-build -b html docs/source docs/build

# 使用pydoc生成文档
python -m pydoc -w interfaces
```
