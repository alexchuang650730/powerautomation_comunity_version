#!/usr/bin/env python3
"""
MCP Coordinator 模拟测试工具

此工具用于模拟MCP Coordinator的行为，测试智能体指令流与适配器的交互。
支持Kilo Code适配器和Gemini适配器的端到端测试。

使用方法：
    python mcp_coordinator_test.py --adapter [kilocode|gemini] --config path/to/config.json
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_coordinator_test")

# 添加适配器路径
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ADAPTER_PATH not in sys.path:
    sys.path.insert(0, ADAPTER_PATH)

# 模拟MCP Coordinator类
class MCPCoordinator:
    """
    模拟MCP Coordinator，用于测试智能体指令流与适配器的交互
    """
    
    def __init__(self):
        """初始化MCP Coordinator"""
        self.adapters = {}
        self.commands_history = []
        logger.info("MCP Coordinator initialized")
    
    def register_adapter(self, name: str, adapter: Any) -> bool:
        """
        注册适配器
        
        Args:
            name: 适配器名称
            adapter: 适配器实例
            
        Returns:
            注册是否成功
        """
        try:
            self.adapters[name] = adapter
            logger.info(f"Adapter '{name}' registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register adapter '{name}': {str(e)}")
            return False
    
    def execute_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行命令
        
        Args:
            command_data: 命令数据
            
        Returns:
            命令执行结果
        """
        try:
            command = command_data.get("command")
            params = command_data.get("params", {})
            adapter_name = command_data.get("adapter", "default")
            
            logger.info(f"Executing command '{command}' with adapter '{adapter_name}'")
            
            # 记录命令历史
            self.commands_history.append({
                "command": command,
                "params": params,
                "adapter": adapter_name,
                "timestamp": time.time()
            })
            
            # 获取适配器
            adapter = self.adapters.get(adapter_name)
            if not adapter:
                return {
                    "status": "error",
                    "message": f"Adapter '{adapter_name}' not found",
                    "result": None
                }
            
            # 执行命令
            result = self._dispatch_command(adapter, command, params)
            
            return {
                "status": "success",
                "message": f"Command '{command}' executed successfully",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {
                "status": "error",
                "message": f"Error executing command: {str(e)}",
                "result": None
            }
    
    def _dispatch_command(self, adapter: Any, command: str, params: Dict[str, Any]) -> Any:
        """
        分发命令到适配器
        
        Args:
            adapter: 适配器实例
            command: 命令名称
            params: 命令参数
            
        Returns:
            命令执行结果
        """
        # 代码生成命令
        if command == "generate_code":
            prompt = params.get("prompt", "")
            mode = params.get("mode", "standard")
            context = params.get("context")
            
            return adapter.generate_code(prompt, context, mode)
        
        # 代码解释命令
        elif command == "interpret_code":
            code = params.get("code", "")
            context = params.get("context")
            
            return adapter.interpret_code(code, context)
        
        # 任务分解命令
        elif command == "decompose_task":
            task = params.get("task", "")
            
            return adapter.decompose_task(task)
        
        # 代码优化命令
        elif command == "optimize_code":
            code = params.get("code", "")
            level = params.get("level", "medium")
            
            return adapter.optimize_code(code, level)
        
        # 复杂度分析命令
        elif command == "analyze_complexity":
            code = params.get("code", "")
            
            return adapter.analyze_complexity(code)
        
        # 健康检查命令
        elif command == "health_check":
            return adapter.health_check()
        
        # 未知命令
        else:
            raise ValueError(f"Unknown command: {command}")
    
    def get_commands_history(self) -> List[Dict[str, Any]]:
        """
        获取命令历史
        
        Returns:
            命令历史列表
        """
        return self.commands_history

# 测试场景
class TestScenario:
    """测试场景基类"""
    
    def __init__(self, coordinator: MCPCoordinator, adapter: Any):
        """
        初始化测试场景
        
        Args:
            coordinator: MCP Coordinator实例
            adapter: 适配器实例
        """
        self.coordinator = coordinator
        self.adapter = adapter
        self.results = []
    
    def run(self) -> List[Dict[str, Any]]:
        """
        运行测试场景
        
        Returns:
            测试结果列表
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def save_results(self, output_path: str) -> None:
        """
        保存测试结果
        
        Args:
            output_path: 输出文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存结果
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Test results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")

# 代码生成测试场景
class CodeGenerationScenario(TestScenario):
    """代码生成测试场景"""
    
    def run(self) -> List[Dict[str, Any]]:
        """
        运行代码生成测试场景
        
        Returns:
            测试结果列表
        """
        logger.info("Running code generation test scenario")
        
        test_cases = [
            {
                "name": "Factorial function",
                "prompt": "Write a Python function to calculate the factorial of a number",
                "mode": "standard"
            },
            {
                "name": "Palindrome check",
                "prompt": "Create a JavaScript function to check if a string is a palindrome",
                "mode": "explained"
            },
            {
                "name": "Binary search",
                "prompt": "Implement a binary search algorithm in C++",
                "mode": "optimized"
            }
        ]
        
        for test_case in test_cases:
            try:
                logger.info(f"Testing code generation: {test_case['name']}")
                
                # 执行命令
                result = self.coordinator.execute_command({
                    "command": "generate_code",
                    "adapter": "code_generation",
                    "params": {
                        "prompt": test_case["prompt"],
                        "mode": test_case["mode"]
                    }
                })
                
                # 记录结果
                self.results.append({
                    "test_case": test_case["name"],
                    "command": "generate_code",
                    "status": result["status"],
                    "code_length": len(result["result"]) if result["status"] == "success" else 0,
                    "success": result["status"] == "success" and len(result["result"]) > 0
                })
                
                logger.info(f"Test case '{test_case['name']}' completed: {result['status']}")
            except Exception as e:
                logger.error(f"Error in test case '{test_case['name']}': {str(e)}")
                self.results.append({
                    "test_case": test_case["name"],
                    "command": "generate_code",
                    "status": "error",
                    "error": str(e),
                    "success": False
                })
        
        return self.results

# 代码解释测试场景
class CodeInterpretationScenario(TestScenario):
    """代码解释测试场景"""
    
    def run(self) -> List[Dict[str, Any]]:
        """
        运行代码解释测试场景
        
        Returns:
            测试结果列表
        """
        logger.info("Running code interpretation test scenario")
        
        test_cases = [
            {
                "name": "Factorial function",
                "code": """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
                """
            },
            {
                "name": "Palindrome check",
                "code": """
function isPalindrome(str) {
    const cleanStr = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    return cleanStr === cleanStr.split('').reverse().join('');
}
                """
            },
            {
                "name": "Binary search",
                "code": """
int binarySearch(int arr[], int l, int r, int x) {
    if (r >= l) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }
    return -1;
}
                """
            }
        ]
        
        for test_case in test_cases:
            try:
                logger.info(f"Testing code interpretation: {test_case['name']}")
                
                # 执行命令
                result = self.coordinator.execute_command({
                    "command": "interpret_code",
                    "adapter": "code_generation",
                    "params": {
                        "code": test_case["code"]
                    }
                })
                
                # 记录结果
                self.results.append({
                    "test_case": test_case["name"],
                    "command": "interpret_code",
                    "status": result["status"],
                    "has_description": "description" in result["result"] if result["status"] == "success" else False,
                    "has_complexity": "complexity" in result["result"] if result["status"] == "success" else False,
                    "success": result["status"] == "success" and isinstance(result["result"], dict) and len(result["result"]) > 0
                })
                
                logger.info(f"Test case '{test_case['name']}' completed: {result['status']}")
            except Exception as e:
                logger.error(f"Error in test case '{test_case['name']}': {str(e)}")
                self.results.append({
                    "test_case": test_case["name"],
                    "command": "interpret_code",
                    "status": "error",
                    "error": str(e),
                    "success": False
                })
        
        return self.results

# 端到端工作流测试场景
class EndToEndWorkflowScenario(TestScenario):
    """端到端工作流测试场景"""
    
    def run(self) -> List[Dict[str, Any]]:
        """
        运行端到端工作流测试场景
        
        Returns:
            测试结果列表
        """
        logger.info("Running end-to-end workflow test scenario")
        
        try:
            # 步骤1：生成代码
            logger.info("Step 1: Generate code")
            generate_result = self.coordinator.execute_command({
                "command": "generate_code",
                "adapter": "code_generation",
                "params": {
                    "prompt": "Write a Python function to find the longest common subsequence of two strings",
                    "mode": "standard"
                }
            })
            
            if generate_result["status"] != "success" or not generate_result["result"]:
                raise ValueError("Failed to generate code")
            
            code = generate_result["result"]
            self.results.append({
                "step": "generate_code",
                "status": generate_result["status"],
                "success": True
            })
            
            # 步骤2：解释代码
            logger.info("Step 2: Interpret code")
            interpret_result = self.coordinator.execute_command({
                "command": "interpret_code",
                "adapter": "code_generation",
                "params": {
                    "code": code
                }
            })
            
            if interpret_result["status"] != "success" or not interpret_result["result"]:
                raise ValueError("Failed to interpret code")
            
            interpretation = interpret_result["result"]
            self.results.append({
                "step": "interpret_code",
                "status": interpret_result["status"],
                "success": True
            })
            
            # 步骤3：优化代码
            logger.info("Step 3: Optimize code")
            optimize_result = self.coordinator.execute_command({
                "command": "optimize_code",
                "adapter": "code_generation",
                "params": {
                    "code": code,
                    "level": "high"
                }
            })
            
            if optimize_result["status"] != "success" or not optimize_result["result"]:
                raise ValueError("Failed to optimize code")
            
            optimized_code = optimize_result["result"]
            self.results.append({
                "step": "optimize_code",
                "status": optimize_result["status"],
                "success": True
            })
            
            # 步骤4：分析复杂度
            logger.info("Step 4: Analyze complexity")
            analyze_result = self.coordinator.execute_command({
                "command": "analyze_complexity",
                "adapter": "code_generation",
                "params": {
                    "code": optimized_code
                }
            })
            
            if analyze_result["status"] != "success" or not analyze_result["result"]:
                raise ValueError("Failed to analyze complexity")
            
            analysis = analyze_result["result"]
            self.results.append({
                "step": "analyze_complexity",
                "status": analyze_result["status"],
                "success": True
            })
            
            # 记录整体结果
            self.results.append({
                "step": "end_to_end_workflow",
                "status": "success",
                "success": True,
                "workflow_completed": True
            })
            
            logger.info("End-to-end workflow test completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in end-to-end workflow test: {str(e)}")
            self.results.append({
                "step": "end_to_end_workflow",
                "status": "error",
                "error": str(e),
                "success": False,
                "workflow_completed": False
            })
            return self.results

# 加载适配器
def load_adapter(adapter_type: str, config_path: str) -> Any:
    """
    加载适配器
    
    Args:
        adapter_type: 适配器类型
        config_path: 配置文件路径
        
    Returns:
        适配器实例
    """
    try:
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载适配器
        if adapter_type == "kilocode":
            from adapters.kilocode.kilocode_adapter import KiloCodeAdapter
            adapter = KiloCodeAdapter()
        elif adapter_type == "gemini":
            from adapters.kilocode.gemini_adapter import GeminiAdapter
            adapter = GeminiAdapter(api_key=config.get("api_key"))
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # 初始化适配器
        if not adapter.initialize(config):
            raise ValueError(f"Failed to initialize {adapter_type} adapter")
        
        logger.info(f"{adapter_type} adapter loaded successfully")
        return adapter
    except Exception as e:
        logger.error(f"Failed to load {adapter_type} adapter: {str(e)}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MCP Coordinator Test")
    parser.add_argument("--adapter", type=str, required=True, choices=["kilocode", "gemini"], help="Adapter type")
    parser.add_argument("--config", type=str, required=True, help="Path to adapter configuration file")
    parser.add_argument("--output", type=str, default="../results/mcp_coordinator_test_results.json", help="Path to output file")
    parser.add_argument("--scenario", type=str, default="all", choices=["code_generation", "code_interpretation", "end_to_end", "all"], help="Test scenario to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 加载适配器
        adapter = load_adapter(args.adapter, args.config)
        
        # 创建MCP Coordinator
        coordinator = MCPCoordinator()
        
        # 注册适配器
        coordinator.register_adapter("code_generation", adapter)
        
        # 运行测试场景
        all_results = []
        
        if args.scenario == "code_generation" or args.scenario == "all":
            scenario = CodeGenerationScenario(coordinator, adapter)
            results = scenario.run()
            all_results.extend(results)
        
        if args.scenario == "code_interpretation" or args.scenario == "all":
            scenario = CodeInterpretationScenario(coordinator, adapter)
            results = scenario.run()
            all_results.extend(results)
        
        if args.scenario == "end_to_end" or args.scenario == "all":
            scenario = EndToEndWorkflowScenario(coordinator, adapter)
            results = scenario.run()
            all_results.extend(results)
        
        # 保存结果
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # 添加时间戳
            final_results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "adapter": args.adapter,
                "scenario": args.scenario,
                "results": all_results
            }
            
            # 计算总体通过率
            passed = sum(1 for result in all_results if result.get("success", False))
            total = len(all_results)
            final_results["overall_pass_rate"] = f"{passed}/{total} ({passed/total*100:.0f}%)" if total > 0 else "N/A"
            
            # 保存结果
            with open(args.output, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
        
        # 输出总结
        passed = sum(1 for result in all_results if result.get("success", False))
        total = len(all_results)
        logger.info(f"Overall test results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        
        # 根据结果设置退出码
        if passed / total >= 0.8:  # 至少80%的测试通过
            logger.info("Tests PASSED")
            sys.exit(0)
        else:
            logger.warning("Tests FAILED")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # 关闭适配器
        try:
            adapter.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()
