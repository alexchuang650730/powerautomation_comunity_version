#!/usr/bin/env python3
"""
模拟Gemini适配器实现

此模块提供了一个模拟的Gemini适配器实现，用于在没有有效API密钥的情况下进行测试。
它模拟了Gemini API的行为，返回预定义的响应，便于进行集成测试。
"""

import os
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Union

# 导入接口定义
from ..interfaces.code_generation_interface import CodeGenerationInterface
from ..interfaces.code_optimization_interface import CodeOptimizationInterface
from ..interfaces.adapter_interface import KiloCodeAdapterInterface

# 配置日志
logging.basicConfig(
    level=os.environ.get("GEMINI_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.environ.get("GEMINI_LOG_FILE", None)
)
logger = logging.getLogger("mock_gemini_adapter")

class MockGeminiAdapter(CodeGenerationInterface, CodeOptimizationInterface, KiloCodeAdapterInterface):
    """
    模拟Gemini适配器实现，提供代码生成、解释、优化等功能的模拟响应。
    
    此适配器不需要实际的API调用，而是返回预定义的响应，用于测试集成流程。
    所有方法都严格遵循接口标准，确保与系统的兼容性。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化模拟Gemini适配器
        
        Args:
            api_key: 模拟API密钥，实际不会使用
        """
        self.api_key = api_key or "mock_api_key_for_testing"
        
        # 初始化模型名称
        self.model_name = "gemini-1.5-pro-mock"
        
        logger.info(f"Initialized Mock Gemini adapter with model: {self.model_name}")
        
        # 初始化能力标志
        self._capabilities = {
            "code_generation": True,
            "code_interpretation": True,
            "task_decomposition": True,
            "code_optimization": True,
            "complexity_analysis": True
        }
        
        # 初始化生成参数
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        # 初始化模拟响应库
        self._initialize_mock_responses()
    
    def _initialize_mock_responses(self):
        """初始化模拟响应库"""
        self.mock_responses = {
            "code_generation": {
                "python": [
                    """
def fibonacci(n):
    """Calculate the Fibonacci number at position n."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    n = 10
    print(f"Fibonacci number at position {n} is {fibonacci(n)}")

if __name__ == "__main__":
    main()
                    """,
                    """
class BinarySearchTree:
    """A simple binary search tree implementation."""
    
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if self.value is None:
            self.value = value
            return
        
        if value < self.value:
            if self.left is None:
                self.left = BinarySearchTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinarySearchTree(value)
            else:
                self.right.insert(value)
    
    def search(self, value):
        if self.value == value:
            return True
        
        if value < self.value:
            if self.left is None:
                return False
            return self.left.search(value)
        else:
            if self.right is None:
                return False
            return self.right.search(value)
                    """
                ],
                "javascript": [
                    """
function quickSort(arr) {
  if (arr.length <= 1) {
    return arr;
  }
  
  const pivot = arr[Math.floor(arr.length / 2)];
  const left = [];
  const middle = [];
  const right = [];
  
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < pivot) {
      left.push(arr[i]);
    } else if (arr[i] > pivot) {
      right.push(arr[i]);
    } else {
      middle.push(arr[i]);
    }
  }
  
  return quickSort(left).concat(middle, quickSort(right));
}

// Example usage
const unsortedArray = [5, 3, 7, 6, 2, 9];
console.log(quickSort(unsortedArray));
                    """
                ]
            },
            "code_interpretation": {
                "simple": {
                    "description": "This code implements a recursive Fibonacci function that calculates the nth Fibonacci number.",
                    "complexity": {
                        "time": "O(2^n)",
                        "space": "O(n)"
                    },
                    "issues": [
                        "Inefficient recursive implementation with exponential time complexity",
                        "No input validation"
                    ],
                    "suggestions": [
                        "Use dynamic programming or iteration to improve time complexity to O(n)",
                        "Add input validation to handle negative numbers and other edge cases"
                    ]
                },
                "complex": {
                    "description": "This code implements a binary search tree with insert and search operations.",
                    "complexity": {
                        "time": "O(log n) average case, O(n) worst case",
                        "space": "O(n)"
                    },
                    "issues": [
                        "No balancing mechanism, could degenerate to O(n) performance",
                        "No deletion operation implemented"
                    ],
                    "suggestions": [
                        "Implement tree balancing (AVL or Red-Black)",
                        "Add deletion operation",
                        "Consider adding traversal methods"
                    ]
                }
            },
            "task_decomposition": [
                [
                    {
                        "id": 1,
                        "description": "Set up project structure and dependencies",
                        "estimated_time": "30m"
                    },
                    {
                        "id": 2,
                        "description": "Implement data models and database schema",
                        "estimated_time": "2h"
                    },
                    {
                        "id": 3,
                        "description": "Create API endpoints for CRUD operations",
                        "estimated_time": "3h"
                    },
                    {
                        "id": 4,
                        "description": "Implement authentication and authorization",
                        "estimated_time": "2h"
                    },
                    {
                        "id": 5,
                        "description": "Write unit and integration tests",
                        "estimated_time": "2h"
                    },
                    {
                        "id": 6,
                        "description": "Set up CI/CD pipeline",
                        "estimated_time": "1h"
                    },
                    {
                        "id": 7,
                        "description": "Deploy to staging environment",
                        "estimated_time": "30m"
                    }
                ]
            ],
            "code_optimization": {
                "low": """
def fibonacci(n):
    """Calculate the Fibonacci number at position n."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
                """,
                "medium": """
def fibonacci(n):
    """Calculate the Fibonacci number at position n using dynamic programming."""
    if n <= 1:
        return n
    
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib[n]
                """,
                "high": """
def fibonacci(n):
    """Calculate the Fibonacci number at position n using optimized iteration."""
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b
                """
            },
            "complexity_analysis": {
                "simple": {
                    "time_complexity": "O(n)",
                    "space_complexity": "O(1)",
                    "details": {
                        "loops": 1,
                        "nested_depth": 1,
                        "recursive": False,
                        "algorithm_type": "linear search",
                        "best_case_time": "O(1)",
                        "average_case_time": "O(n/2)",
                        "worst_case_time": "O(n)",
                        "stability": "N/A",
                        "in_place": True
                    }
                },
                "complex": {
                    "time_complexity": "O(n log n)",
                    "space_complexity": "O(n)",
                    "details": {
                        "loops": 2,
                        "nested_depth": 2,
                        "recursive": True,
                        "algorithm_type": "divide and conquer",
                        "best_case_time": "O(n log n)",
                        "average_case_time": "O(n log n)",
                        "worst_case_time": "O(n log n)",
                        "stability": False,
                        "in_place": False
                    }
                }
            },
            "improvements": [
                [
                    {
                        "type": "performance",
                        "description": "Replace recursive implementation with iterative approach to avoid stack overflow and improve performance",
                        "code_snippet": """
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
                        """
                    },
                    {
                        "type": "readability",
                        "description": "Add proper docstring and type hints",
                        "code_snippet": """
def fibonacci(n: int) -> int:
    \"\"\"
    Calculate the Fibonacci number at position n.
    
    Args:
        n: A non-negative integer position in the Fibonacci sequence
        
    Returns:
        The Fibonacci number at position n
        
    Raises:
        ValueError: If n is negative
    \"\"\"
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
                        """
                    }
                ]
            ]
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        使用配置信息初始化适配器
        
        Args:
            config: 配置信息字典
            
        Returns:
            初始化是否成功
        """
        try:
            # 更新配置
            if "api_key" in config:
                self.api_key = config["api_key"]
            
            if "model" in config:
                self.model_name = config["model"] + "-mock"
            
            if "temperature" in config:
                self.generation_config["temperature"] = config["temperature"]
            
            if "max_output_tokens" in config:
                self.generation_config["max_output_tokens"] = config["max_output_tokens"]
            
            if "capabilities" in config:
                self._capabilities.update(config["capabilities"])
            
            logger.info("Mock Gemini adapter initialized successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error initializing Mock Gemini adapter: {str(e)}")
            return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        获取适配器支持的能力
        
        Returns:
            支持的能力字典，键为能力名称，值为是否支持
        """
        return self._capabilities.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """
        检查适配器健康状态
        
        Returns:
            健康状态信息字典
        """
        return {
            "status": "ok",
            "message": "Mock Gemini service is healthy",
            "details": {
                "model": self.model_name,
                "response": "This is a mock response from the Gemini API"
            }
        }
    
    def shutdown(self) -> bool:
        """
        关闭适配器，释放资源
        
        Returns:
            关闭是否成功
        """
        logger.info("Mock Gemini adapter shut down successfully")
        return True
    
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
            
        Raises:
            ValueError: 如果提示为空或模式无效
            RuntimeError: 如果API调用失败
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        if mode not in ["standard", "optimized", "explained"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: standard, optimized, explained")
        
        # 模拟API延迟
        time.sleep(0.5)
        
        # 根据提示关键词选择语言
        language = "python"
        if any(keyword in prompt.lower() for keyword in ["javascript", "js", "node", "react", "vue"]):
            language = "javascript"
        
        # 从模拟响应库中随机选择一个响应
        responses = self.mock_responses["code_generation"].get(language, self.mock_responses["code_generation"]["python"])
        code = random.choice(responses)
        
        return code.strip()
    
    def interpret_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        解释代码
        
        Args:
            code: 需要解释的代码
            context: 上下文信息
            
        Returns:
            包含代码解释的字典
            
        Raises:
            ValueError: 如果代码为空
            RuntimeError: 如果API调用失败
        """
        if not code:
            raise ValueError("Code cannot be empty")
        
        # 模拟API延迟
        time.sleep(0.7)
        
        # 根据代码长度选择响应
        if len(code) > 500:
            return self.mock_responses["code_interpretation"]["complex"]
        else:
            return self.mock_responses["code_interpretation"]["simple"]
    
    def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        分解任务
        
        Args:
            task_description: 任务描述
            
        Returns:
            分解后的子任务列表
            
        Raises:
            ValueError: 如果任务描述为空
            RuntimeError: 如果API调用失败
        """
        if not task_description:
            raise ValueError("Task description cannot be empty")
        
        # 模拟API延迟
        time.sleep(0.6)
        
        # 返回预定义的任务分解
        return random.choice(self.mock_responses["task_decomposition"])
    
    def batch_generate(self, prompts: List[str], context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        批量生成代码
        
        Args:
            prompts: 代码生成提示列表
            context: 共享上下文信息
            
        Returns:
            生成的代码字符串列表
            
        Raises:
            ValueError: 如果提示列表为空
            RuntimeError: 如果API调用失败
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        results = []
        
        for prompt in prompts:
            try:
                code = self.generate_code(prompt, context)
                results.append(code)
            except Exception as e:
                logger.error(f"Error generating code for prompt '{prompt[:30]}...': {str(e)}")
                results.append("")  # 添加空字符串作为占位符
        
        return results
    
    def optimize_code(self, code: str, optimization_level: str = "medium") -> str:
        """
        优化代码
        
        Args:
            code: 需要优化的代码
            optimization_level: 优化级别，可选值包括 "low", "medium", "high"
            
        Returns:
            优化后的代码
            
        Raises:
            ValueError: 如果代码为空或优化级别无效
            RuntimeError: 如果API调用失败
        """
        if not code:
            raise ValueError("Code cannot be empty")
        
        if optimization_level not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid optimization level: {optimization_level}. Must be one of: low, medium, high")
        
        # 模拟API延迟
        time.sleep(0.8)
        
        # 返回预定义的优化代码
        return self.mock_responses["code_optimization"][optimization_level].strip()
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """
        分析代码复杂度
        
        Args:
            code: 需要分析的代码
            
        Returns:
            包含复杂度分析的字典
            
        Raises:
            ValueError: 如果代码为空
            RuntimeError: 如果API调用失败
        """
        if not code:
            raise ValueError("Code cannot be empty")
        
        # 模拟API延迟
        time.sleep(0.5)
        
        # 根据代码长度选择响应
        if len(code) > 500:
            return self.mock_responses["complexity_analysis"]["complex"]
        else:
            return self.mock_responses["complexity_analysis"]["simple"]
    
    def suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """
        提供代码改进建议
        
        Args:
            code: 需要改进的代码
            
        Returns:
            改进建议列表
            
        Raises:
            ValueError: 如果代码为空
            RuntimeError: 如果API调用失败
        """
        if not code:
            raise ValueError("Code cannot be empty")
        
        # 模拟API延迟
        time.sleep(0.7)
        
        # 返回预定义的改进建议
        return random.choice(self.mock_responses["improvements"])
