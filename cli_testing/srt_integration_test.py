#!/usr/bin/env python3
"""
SRT适配器系统级端到端测试工具

此工具用于测试SRT适配器与MCP Coordinator的集成流程，
验证自我奖励训练、评估和改进功能在端到端场景中的表现。
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("srt_integration_test")

# 添加父目录到路径，以便导入适配器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入SRT适配器和集成模块
from adapters.srt.srt_adapter import SRTAdapter
from integration.rlfactory_srt_integration import RLFactorySRTIntegration

class MCPCoordinatorSRTTest:
    """模拟MCP Coordinator的SRT适配器测试类"""
    
    def __init__(self, config_path: str, output_dir: str = "results", verbose: bool = False):
        """
        初始化测试环境
        
        Args:
            config_path: SRT配置文件路径
            output_dir: 测试结果输出目录
            verbose: 是否输出详细日志
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.verbose = verbose
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化适配器和集成模块
        self.srt_adapter = None
        self.rl_factory = None
        self.config = self._load_config()
        
        # 设置日志级别
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}
    
    def initialize(self) -> bool:
        """初始化测试环境"""
        try:
            # 初始化SRT适配器
            logger.info("Initializing SRT adapter...")
            self.srt_adapter = SRTAdapter()
            if not self.srt_adapter.initialize(self.config):
                logger.error("Failed to initialize SRT adapter")
                return False
            
            # 初始化RL Factory集成
            logger.info("Initializing RL Factory integration...")
            self.rl_factory = RLFactorySRTIntegration(self.srt_adapter)
            
            logger.info("Test environment initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize test environment: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试场景"""
        results = {
            "training": self.test_training(),
            "evaluation": self.test_evaluation(),
            "improvement": self.test_improvement(),
            "end_to_end": self.test_end_to_end(),
            "batch_training": self.test_batch_training(),
            "model_save_load": self.test_model_save_load()
        }
        
        # 计算总体通过率
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result["status"] == "success")
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": f"{passed_tests / total_tests * 100:.2f}%"
        }
        
        # 保存测试结果
        self._save_results(results)
        
        return results
    
    def test_training(self) -> Dict[str, Any]:
        """测试训练功能"""
        logger.info("Running training test...")
        try:
            # 准备训练数据
            training_data = self._prepare_training_data()
            
            # 执行训练
            start_time = time.time()
            training_result = self.rl_factory.train(
                training_data["prompt"],
                training_data["context"],
                training_data["expected_output"]
            )
            duration = time.time() - start_time
            
            # 验证结果
            success = self._validate_training_result(training_result)
            
            result = {
                "status": "success" if success else "failure",
                "duration": f"{duration:.2f}s",
                "details": training_result
            }
            
            logger.info(f"Training test completed: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"Training test failed: {str(e)}")
            return {
                "status": "failure",
                "error": str(e)
            }
    
    def test_evaluation(self) -> Dict[str, Any]:
        """测试评估功能"""
        logger.info("Running evaluation test...")
        try:
            # 准备评估数据
            eval_data = self._prepare_evaluation_data()
            
            # 执行评估
            start_time = time.time()
            eval_result = self.rl_factory.evaluate(
                eval_data["prompt"],
                eval_data["context"],
                eval_data["response"],
                eval_data["expected_output"]
            )
            duration = time.time() - start_time
            
            # 验证结果
            success = self._validate_evaluation_result(eval_result)
            
            result = {
                "status": "success" if success else "failure",
                "duration": f"{duration:.2f}s",
                "details": eval_result
            }
            
            logger.info(f"Evaluation test completed: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"Evaluation test failed: {str(e)}")
            return {
                "status": "failure",
                "error": str(e)
            }
    
    def test_improvement(self) -> Dict[str, Any]:
        """测试改进功能"""
        logger.info("Running improvement test...")
        try:
            # 准备改进数据
            improve_data = self._prepare_improvement_data()
            
            # 执行改进
            start_time = time.time()
            improve_result = self.rl_factory.improve(
                improve_data["prompt"],
                improve_data["context"],
                improve_data["response"],
                improve_data["feedback"]
            )
            duration = time.time() - start_time
            
            # 验证结果
            success = self._validate_improvement_result(improve_result)
            
            result = {
                "status": "success" if success else "failure",
                "duration": f"{duration:.2f}s",
                "details": improve_result
            }
            
            logger.info(f"Improvement test completed: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"Improvement test failed: {str(e)}")
            return {
                "status": "failure",
                "error": str(e)
            }
    
    def test_end_to_end(self) -> Dict[str, Any]:
        """测试端到端流程"""
        logger.info("Running end-to-end test...")
        try:
            # 准备端到端测试数据
            e2e_data = self._prepare_end_to_end_data()
            
            # 第1步：训练
            logger.info("Step 1: Training model...")
            training_result = self.rl_factory.train(
                e2e_data["prompt"],
                e2e_data["context"],
                e2e_data["expected_output"]
            )
            
            # 第2步：生成响应
            logger.info("Step 2: Generating response...")
            response = "This is a simulated response from the model"
            
            # 第3步：评估响应
            logger.info("Step 3: Evaluating response...")
            eval_result = self.rl_factory.evaluate(
                e2e_data["prompt"],
                e2e_data["context"],
                response,
                e2e_data["expected_output"]
            )
            
            # 第4步：基于评估结果改进
            logger.info("Step 4: Improving model based on evaluation...")
            improve_result = self.rl_factory.improve(
                e2e_data["prompt"],
                e2e_data["context"],
                response,
                f"Reward score: {eval_result['reward']}, needs improvement in accuracy"
            )
            
            # 验证端到端流程
            success = (
                self._validate_training_result(training_result) and
                self._validate_evaluation_result(eval_result) and
                self._validate_improvement_result(improve_result)
            )
            
            result = {
                "status": "success" if success else "failure",
                "details": {
                    "training": training_result,
                    "evaluation": eval_result,
                    "improvement": improve_result
                }
            }
            
            logger.info(f"End-to-end test completed: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"End-to-end test failed: {str(e)}")
            return {
                "status": "failure",
                "error": str(e)
            }
    
    def test_batch_training(self) -> Dict[str, Any]:
        """测试批量训练功能"""
        logger.info("Running batch training test...")
        try:
            # 准备批量训练数据
            batch_data = self._prepare_batch_training_data()
            
            # 执行批量训练
            start_time = time.time()
            batch_result = self.rl_factory.batch_train(
                batch_data["prompts"],
                batch_data["contexts"],
                batch_data["expected_outputs"]
            )
            duration = time.time() - start_time
            
            # 验证结果
            success = self._validate_batch_training_result(batch_result)
            
            result = {
                "status": "success" if success else "failure",
                "duration": f"{duration:.2f}s",
                "details": batch_result
            }
            
            logger.info(f"Batch training test completed: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"Batch training test failed: {str(e)}")
            return {
                "status": "failure",
                "error": str(e)
            }
    
    def test_model_save_load(self) -> Dict[str, Any]:
        """测试模型保存和加载功能"""
        logger.info("Running model save/load test...")
        try:
            # 保存模型
            model_path = os.path.join(self.output_dir, "test_model.pt")
            logger.info(f"Saving model to {model_path}...")
            save_result = self.rl_factory.save_model(model_path)
            
            # 加载模型
            logger.info(f"Loading model from {model_path}...")
            load_result = self.rl_factory.load_model(model_path)
            
            # 验证结果
            success = save_result and load_result
            
            result = {
                "status": "success" if success else "failure",
                "details": {
                    "save_result": save_result,
                    "load_result": load_result,
                    "model_path": model_path
                }
            }
            
            logger.info(f"Model save/load test completed: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"Model save/load test failed: {str(e)}")
            return {
                "status": "failure",
                "error": str(e)
            }
    
    def _prepare_training_data(self) -> Dict[str, Any]:
        """准备训练数据"""
        return {
            "prompt": "Write a function to calculate the factorial of a number",
            "context": {"language": "python", "complexity": "medium"},
            "expected_output": """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
"""
        }
    
    def _prepare_evaluation_data(self) -> Dict[str, Any]:
        """准备评估数据"""
        return {
            "prompt": "Write a function to check if a string is a palindrome",
            "context": {"language": "python", "case_sensitive": False},
            "response": """
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]
""",
            "expected_output": """
def is_palindrome(s):
    if not s:
        return True
    s = s.lower()
    return s == s[::-1]
"""
        }
    
    def _prepare_improvement_data(self) -> Dict[str, Any]:
        """准备改进数据"""
        return {
            "prompt": "Write a function to find the maximum element in a binary tree",
            "context": {"language": "python"},
            "response": """
def find_max(root):
    if not root:
        return float('-inf')
    return max(root.val, find_max(root.left), find_max(root.right))
""",
            "feedback": "The function is correct but inefficient for large trees. Consider adding memoization."
        }
    
    def _prepare_end_to_end_data(self) -> Dict[str, Any]:
        """准备端到端测试数据"""
        return {
            "prompt": "Write a function to merge two sorted arrays",
            "context": {"language": "python", "optimize": True},
            "expected_output": """
def merge_sorted_arrays(arr1, arr2):
    result = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result
"""
        }
    
    def _prepare_batch_training_data(self) -> Dict[str, Any]:
        """准备批量训练数据"""
        return {
            "prompts": [
                "Write a function to calculate the factorial of a number",
                "Write a function to check if a string is a palindrome",
                "Write a function to find the nth Fibonacci number"
            ],
            "contexts": [
                {"language": "python"},
                {"language": "python", "case_sensitive": False},
                {"language": "python", "optimize": True}
            ],
            "expected_outputs": [
                "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
                "def is_palindrome(s):\n    if not s:\n        return True\n    s = s.lower()\n    return s == s[::-1]",
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
            ]
        }
    
    def _validate_training_result(self, result: Dict[str, Any]) -> bool:
        """验证训练结果"""
        if not isinstance(result, dict):
            return False
        
        required_keys = ["model_state", "reward", "iterations"]
        if not all(key in result for key in required_keys):
            return False
        
        # 验证奖励值是否合理
        if not isinstance(result["reward"], (int, float)) or result["reward"] < 0:
            return False
        
        return True
    
    def _validate_evaluation_result(self, result: Dict[str, Any]) -> bool:
        """验证评估结果"""
        if not isinstance(result, dict):
            return False
        
        required_keys = ["reward", "metrics"]
        if not all(key in result for key in required_keys):
            return False
        
        # 验证奖励值是否合理
        if not isinstance(result["reward"], (int, float)) or result["reward"] < 0:
            return False
        
        return True
    
    def _validate_improvement_result(self, result: Dict[str, Any]) -> bool:
        """验证改进结果"""
        if not isinstance(result, dict):
            return False
        
        required_keys = ["improved_model_state", "improvement_metrics"]
        if not all(key in result for key in required_keys):
            return False
        
        return True
    
    def _validate_batch_training_result(self, result: List[Dict[str, Any]]) -> bool:
        """验证批量训练结果"""
        if not isinstance(result, list):
            return False
        
        # 验证每个训练结果
        for item in result:
            if not self._validate_training_result(item):
                return False
        
        return True
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存测试结果"""
        output_path = os.path.join(self.output_dir, "srt_integration_test_results.json")
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Test results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SRT Adapter Integration Test")
    parser.add_argument("--config", required=True, help="Path to SRT configuration file")
    parser.add_argument("--output", default="results", help="Output directory for test results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # 运行测试
    tester = MCPCoordinatorSRTTest(args.config, args.output, args.verbose)
    if tester.initialize():
        results = tester.run_all_tests()
        
        # 打印测试摘要
        summary = results["summary"]
        print("\n" + "=" * 50)
        print(f"SRT Integration Test Summary:")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed tests: {summary['passed_tests']}")
        print(f"Pass rate: {summary['pass_rate']}")
        print("=" * 50)
        
        # 返回测试结果
        return 0 if summary['passed_tests'] == summary['total_tests'] else 1
    else:
        logger.error("Failed to initialize test environment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
