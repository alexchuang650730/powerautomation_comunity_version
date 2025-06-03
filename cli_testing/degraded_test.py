#!/usr/bin/env python3
"""
降级测试工具 - 适配器功能验证

此工具用于在资源受限环境下验证Kilo Code适配器功能，
通过模拟方式验证SRT适配器接口，不依赖PyTorch。
"""

import os
import sys
import json
import argparse
import logging
import importlib.util
from typing import Dict, Any, List, Optional, Union
import time

# 添加父目录到sys.path，解决相对导入问题
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("degraded_test")

class MockSRTAdapter:
    """
    SRT适配器的模拟实现，不依赖PyTorch
    用于验证接口和集成功能
    """
    
    def __init__(self):
        """初始化模拟适配器"""
        self._capabilities = {
            "self_reward_training": True,
            "thought_evaluation": True,
            "thought_improvement": True,
            "batch_training": True
        }
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """模拟初始化"""
        logger.info("Initializing mock SRT adapter")
        self.initialized = True
        return True
    
    def get_capabilities(self) -> Dict[str, bool]:
        """获取支持的能力"""
        return self._capabilities.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "ok",
            "message": "Mock SRT adapter is healthy",
            "details": {
                "mock": True,
                "initialized": self.initialized
            }
        }
    
    def train(self, thought_process: Union[str, Dict[str, Any]], iterations: int = 100) -> Dict[str, Any]:
        """模拟训练"""
        return {
            "iterations": iterations,
            "improvements": [
                {"iteration": 0, "reward": 0.5, "loss": 0.5},
                {"iteration": iterations-1, "reward": 0.8, "loss": 0.2}
            ],
            "final_reward": 0.8
        }
    
    def evaluate(self, thought_process: Union[str, Dict[str, Any]]) -> float:
        """模拟评估"""
        return 0.75
    
    def improve(self, thought_process: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """模拟改进"""
        if isinstance(thought_process, dict):
            improved = thought_process.copy()
            improved["improved"] = True
            improved["quality_score"] = 0.8
            return improved
        else:
            return f"Improved: {thought_process}\nQuality Score: 0.8"
    
    def batch_train(self, thought_processes: List[Union[str, Dict[str, Any]]], batch_size: int = 32) -> Dict[str, Any]:
        """模拟批量训练"""
        return {
            "batch_size": batch_size,
            "samples_processed": len(thought_processes),
            "average_reward": 0.75
        }
    
    def save_model(self, path: str) -> bool:
        """模拟保存模型"""
        logger.info(f"Mock saving model to {path}")
        return True
    
    def load_model(self, path: str) -> bool:
        """模拟加载模型"""
        logger.info(f"Mock loading model from {path}")
        return True
    
    def shutdown(self) -> bool:
        """模拟关闭"""
        logger.info("Shutting down mock SRT adapter")
        self.initialized = False
        return True

def load_adapter(adapter_type: str):
    """
    加载适配器
    
    Args:
        adapter_type: 适配器类型，'kilocode'或'srt'
        
    Returns:
        适配器实例
    """
    try:
        # 根据适配器类型导入相应模块
        if adapter_type.lower() == 'kilocode':
            from adapters.kilocode.kilocode_adapter import KiloCodeAdapter
            adapter = KiloCodeAdapter()
            logger.info(f"Successfully loaded KiloCode adapter")
        elif adapter_type.lower() == 'srt':
            # 使用模拟适配器代替真实SRT适配器
            logger.info(f"Using mock SRT adapter due to PyTorch dependency limitations")
            adapter = MockSRTAdapter()
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
        
        return adapter
        
    except Exception as e:
        logger.error(f"Failed to load adapter: {str(e)}")
        raise

def format_output(result: Any) -> str:
    """
    格式化输出结果
    
    Args:
        result: 任意结果对象
        
    Returns:
        格式化后的字符串
    """
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2, ensure_ascii=False)
    else:
        return str(result)

def execute_command(adapter, command: str, args: List[str]) -> Any:
    """
    执行命令
    
    Args:
        adapter: 适配器实例
        command: 命令名称
        args: 命令参数
        
    Returns:
        命令执行结果
    """
    # 检查适配器是否支持该命令
    if not hasattr(adapter, command):
        raise AttributeError(f"Adapter does not support command: {command}")
    
    # 获取命令方法
    method = getattr(adapter, command)
    
    # 解析参数
    parsed_args = []
    kwargs = {}
    
    for arg in args:
        if '=' in arg:
            # 关键字参数
            key, value = arg.split('=', 1)
            # 尝试解析JSON
            try:
                kwargs[key] = json.loads(value)
            except json.JSONDecodeError:
                kwargs[key] = value
        else:
            # 位置参数
            # 尝试解析JSON
            try:
                parsed_args.append(json.loads(arg))
            except json.JSONDecodeError:
                parsed_args.append(arg)
    
    # 执行命令
    start_time = time.time()
    result = method(*parsed_args, **kwargs)
    end_time = time.time()
    
    logger.info(f"Command '{command}' executed in {end_time - start_time:.2f} seconds")
    
    return result

def interactive_mode(adapter):
    """
    交互式模式
    
    Args:
        adapter: 适配器实例
    """
    print(f"=== Degraded Test Interactive Mode ===")
    print(f"Type 'help' for available commands, 'exit' to quit")
    
    # 获取适配器支持的命令
    commands = [method for method in dir(adapter) if not method.startswith('_') and callable(getattr(adapter, method))]
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n> ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive mode")
                break
            
            # 检查帮助命令
            if user_input.lower() in ['help', '?']:
                print("\nAvailable commands:")
                for cmd in sorted(commands):
                    doc = getattr(adapter, cmd).__doc__
                    if doc:
                        # 提取第一行作为简短描述
                        desc = doc.strip().split('\n')[0]
                    else:
                        desc = "No description available"
                    print(f"  {cmd}: {desc}")
                continue
            
            # 解析命令和参数
            parts = user_input.split()
            if not parts:
                continue
                
            command = parts[0]
            args = parts[1:]
            
            # 执行命令
            result = execute_command(adapter, command, args)
            
            # 显示结果
            print("\nResult:")
            print(format_output(result))
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
            
        except Exception as e:
            print(f"\nError: {str(e)}")

def test_kilocode_adapter():
    """测试Kilo Code适配器的基本功能"""
    try:
        print("\n=== Testing Kilo Code Adapter ===")
        
        # 加载适配器
        adapter = load_adapter('kilocode')
        
        # 测试健康检查
        print("\nTesting health_check()...")
        health = adapter.health_check()
        print(format_output(health))
        
        # 测试获取能力
        print("\nTesting get_capabilities()...")
        capabilities = adapter.get_capabilities()
        print(format_output(capabilities))
        
        # 测试初始化
        print("\nTesting initialize()...")
        config = {
            "api_key": "test_key",
            "server_url": "https://api.kilocode.ai/v1",
            "timeout": 30
        }
        init_result = adapter.initialize(config)
        print(f"Initialize result: {init_result}")
        
        # 测试代码生成
        print("\nTesting generate_code()...")
        try:
            code = adapter.generate_code("Write a Python function to calculate factorial")
            print(code[:200] + "..." if len(code) > 200 else code)
        except Exception as e:
            print(f"Error in generate_code: {str(e)}")
        
        # 测试关闭
        print("\nTesting shutdown()...")
        shutdown_result = adapter.shutdown()
        print(f"Shutdown result: {shutdown_result}")
        
        print("\nKilo Code Adapter tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing Kilo Code adapter: {str(e)}")
        return False

def test_mock_srt_adapter():
    """测试模拟SRT适配器的基本功能"""
    try:
        print("\n=== Testing Mock SRT Adapter ===")
        
        # 加载适配器
        adapter = load_adapter('srt')
        
        # 测试健康检查
        print("\nTesting health_check()...")
        health = adapter.health_check()
        print(format_output(health))
        
        # 测试获取能力
        print("\nTesting get_capabilities()...")
        capabilities = adapter.get_capabilities()
        print(format_output(capabilities))
        
        # 测试初始化
        print("\nTesting initialize()...")
        config = {
            "model_path": "",
            "batch_size": 16,
            "learning_rate": 0.001
        }
        init_result = adapter.initialize(config)
        print(f"Initialize result: {init_result}")
        
        # 测试评估
        print("\nTesting evaluate()...")
        thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
        score = adapter.evaluate(thought_process)
        print(f"Evaluation score: {score}")
        
        # 测试关闭
        print("\nTesting shutdown()...")
        shutdown_result = adapter.shutdown()
        print(f"Shutdown result: {shutdown_result}")
        
        print("\nMock SRT Adapter tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing mock SRT adapter: {str(e)}")
        return False

def test_integration():
    """测试集成功能"""
    try:
        print("\n=== Testing Integration ===")
        
        # 导入集成模块
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        # 测试MCP Tool与Kilo Code集成
        print("\nTesting MCP Tool - Kilo Code Integration...")
        try:
            from integration.mcptool_kilocode_integration import MCPToolKiloCodeIntegration
            kilocode_integration = MCPToolKiloCodeIntegration()
            print(f"Integration initialized: {kilocode_integration.is_initialized()}")
            
            # 测试基本功能
            capabilities = kilocode_integration.get_capabilities()
            print(f"Integration capabilities: {format_output(capabilities)}")
        except Exception as e:
            print(f"Error in Kilo Code integration: {str(e)}")
        
        # 测试RL Factory与SRT集成
        print("\nTesting RL Factory - SRT Integration...")
        try:
            from integration.rlfactory_srt_integration import RLFactorySRTIntegration
            
            # 修改导入以使用模拟适配器
            import sys
            import types
            
            # 创建模拟模块
            mock_srt_module = types.ModuleType('srt.srt_adapter')
            mock_srt_module.SRTAdapter = MockSRTAdapter
            sys.modules['srt.srt_adapter'] = mock_srt_module
            
            srt_integration = RLFactorySRTIntegration()
            print(f"Integration initialized: {srt_integration.is_initialized()}")
            
            # 测试基本功能
            capabilities = srt_integration.get_capabilities()
            print(f"Integration capabilities: {format_output(capabilities)}")
        except Exception as e:
            print(f"Error in SRT integration: {str(e)}")
        
        print("\nIntegration tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing integration: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Degraded Testing Tool")
    
    parser.add_argument('--adapter', '-a', choices=['kilocode', 'srt', 'all'],
                        help="Adapter type to test")
    
    parser.add_argument('--test', '-t', action='store_true',
                        help="Run automated tests")
    
    parser.add_argument('--integration', '-i', action='store_true',
                        help="Test integration")
    
    parser.add_argument('--command', '-c',
                        help="Command to execute (requires --adapter)")
    
    parser.add_argument('--args', '-g', nargs='*', default=[],
                        help="Arguments for the command")
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 运行自动测试
        if args.test:
            if args.adapter == 'kilocode' or args.adapter == 'all':
                test_kilocode_adapter()
            
            if args.adapter == 'srt' or args.adapter == 'all':
                test_mock_srt_adapter()
            
            if args.integration:
                test_integration()
            
            return 0
        
        # 执行单个命令
        if args.command:
            if not args.adapter or args.adapter == 'all':
                print("Error: --adapter must be specified when using --command")
                return 1
            
            # 加载适配器
            adapter = load_adapter(args.adapter)
            
            # 执行命令
            result = execute_command(adapter, args.command, args.args)
            
            # 输出结果
            print(format_output(result))
            return 0
        
        # 交互式模式
        if args.adapter and args.adapter != 'all':
            adapter = load_adapter(args.adapter)
            interactive_mode(adapter)
            return 0
        
        # 如果没有指定操作，显示帮助
        parser.print_help()
        return 1
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
