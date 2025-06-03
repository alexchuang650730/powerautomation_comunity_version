#!/usr/bin/env python3
"""
真实SRT适配器测试工具

此工具用于测试基于PyTorch的真实SRT适配器功能，
不使用模拟实现，确保验证实际功能。
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
logger = logging.getLogger("real_srt_test")

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
            # 使用真实SRT适配器，不使用模拟实现
            from adapters.srt.srt_adapter import SRTAdapter
            adapter = SRTAdapter()
            logger.info(f"Successfully loaded real SRT adapter")
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
    print(f"=== Real SRT Adapter Interactive Mode ===")
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

def test_srt_adapter():
    """测试真实SRT适配器的基本功能"""
    try:
        print("\n=== Testing Real SRT Adapter ===")
        
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
            "learning_rate": 0.001,
            "device": "cpu"
        }
        init_result = adapter.initialize(config)
        print(f"Initialize result: {init_result}")
        
        # 测试评估
        print("\nTesting evaluate()...")
        try:
            thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
            score = adapter.evaluate(thought_process)
            print(f"Evaluation score: {score}")
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
        
        # 测试训练
        print("\nTesting train()...")
        try:
            thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
            train_result = adapter.train(thought_process, iterations=10)
            print(f"Training result: {format_output(train_result)}")
        except Exception as e:
            print(f"Error in train: {str(e)}")
        
        # 测试改进
        print("\nTesting improve()...")
        try:
            thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
            improved = adapter.improve(thought_process)
            print(f"Improved thought process: {improved[:200]}..." if len(str(improved)) > 200 else improved)
        except Exception as e:
            print(f"Error in improve: {str(e)}")
        
        # 测试批量训练
        print("\nTesting batch_train()...")
        try:
            thought_processes = [
                "I need to solve this problem by breaking it down into steps.",
                "Let me analyze this problem from multiple perspectives."
            ]
            batch_result = adapter.batch_train(thought_processes, batch_size=2)
            print(f"Batch training result: {format_output(batch_result)}")
        except Exception as e:
            print(f"Error in batch_train: {str(e)}")
        
        # 测试保存模型
        print("\nTesting save_model()...")
        try:
            save_path = "/tmp/srt_model_test.pt"
            save_result = adapter.save_model(save_path)
            print(f"Save model result: {save_result}")
        except Exception as e:
            print(f"Error in save_model: {str(e)}")
        
        # 测试加载模型
        print("\nTesting load_model()...")
        try:
            load_path = "/tmp/srt_model_test.pt"
            if os.path.exists(load_path):
                load_result = adapter.load_model(load_path)
                print(f"Load model result: {load_result}")
            else:
                print(f"Model file not found: {load_path}")
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
        
        # 测试关闭
        print("\nTesting shutdown()...")
        shutdown_result = adapter.shutdown()
        print(f"Shutdown result: {shutdown_result}")
        
        print("\nReal SRT Adapter tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing real SRT adapter: {str(e)}")
        return False

def test_integration():
    """测试集成功能"""
    try:
        print("\n=== Testing Integration ===")
        
        # 导入集成模块
        from integration.rlfactory_srt_integration import RLFactorySRTIntegration
        
        # 测试RL Factory与SRT集成
        print("\nTesting RL Factory - SRT Integration...")
        srt_integration = RLFactorySRTIntegration()
        print(f"Integration initialized: {srt_integration.is_initialized()}")
        
        # 测试初始化
        print("\nTesting integration initialize()...")
        config = {
            "model_path": "",
            "batch_size": 16,
            "learning_rate": 0.001,
            "device": "cpu"
        }
        init_result = srt_integration._initialize_adapter()
        print(f"Initialize result: {init_result}")
        
        # 测试获取能力
        print("\nTesting integration get_capabilities()...")
        capabilities = srt_integration.get_capabilities()
        print(f"Integration capabilities: {format_output(capabilities)}")
        
        # 测试训练
        print("\nTesting integration train()...")
        try:
            thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
            train_result = srt_integration.train(thought_process, iterations=10)
            print(f"Training result: {format_output(train_result)}")
        except Exception as e:
            print(f"Error in integration train: {str(e)}")
        
        # 测试评估
        print("\nTesting integration evaluate()...")
        try:
            thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
            eval_result = srt_integration.evaluate(thought_process)
            print(f"Evaluation result: {format_output(eval_result)}")
        except Exception as e:
            print(f"Error in integration evaluate: {str(e)}")
        
        # 测试改进
        print("\nTesting integration improve()...")
        try:
            thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
            improve_result = srt_integration.improve(thought_process)
            print(f"Improvement result: {format_output(improve_result)}")
        except Exception as e:
            print(f"Error in integration improve: {str(e)}")
        
        # 测试批量训练
        print("\nTesting integration batch_train()...")
        try:
            thought_processes = [
                "I need to solve this problem by breaking it down into steps.",
                "Let me analyze this problem from multiple perspectives."
            ]
            batch_result = srt_integration.batch_train(thought_processes, batch_size=2)
            print(f"Batch training result: {format_output(batch_result)}")
        except Exception as e:
            print(f"Error in integration batch_train: {str(e)}")
        
        # 测试保存模型
        print("\nTesting integration save_model()...")
        try:
            save_path = "/tmp/srt_integration_model_test.pt"
            save_result = srt_integration.save_model(save_path)
            print(f"Save model result: {format_output(save_result)}")
        except Exception as e:
            print(f"Error in integration save_model: {str(e)}")
        
        # 测试加载模型
        print("\nTesting integration load_model()...")
        try:
            load_path = "/tmp/srt_integration_model_test.pt"
            if os.path.exists(load_path):
                load_result = srt_integration.load_model(load_path)
                print(f"Load model result: {format_output(load_result)}")
            else:
                print(f"Model file not found: {load_path}")
        except Exception as e:
            print(f"Error in integration load_model: {str(e)}")
        
        # 测试关闭
        print("\nTesting integration shutdown()...")
        shutdown_result = srt_integration.shutdown()
        print(f"Shutdown result: {shutdown_result}")
        
        print("\nIntegration tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing integration: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Real SRT Adapter Testing Tool")
    
    parser.add_argument('--test', '-t', action='store_true',
                        help="Run automated tests for real SRT adapter")
    
    parser.add_argument('--integration', '-i', action='store_true',
                        help="Test integration with RL Factory")
    
    parser.add_argument('--command', '-c',
                        help="Command to execute on SRT adapter")
    
    parser.add_argument('--args', '-g', nargs='*', default=[],
                        help="Arguments for the command")
    
    parser.add_argument('--interactive', '-n', action='store_true',
                        help="Start interactive mode")
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 运行自动测试
        if args.test:
            test_srt_adapter()
        
        # 测试集成
        if args.integration:
            test_integration()
        
        # 执行单个命令
        if args.command:
            adapter = load_adapter('srt')
            result = execute_command(adapter, args.command, args.args)
            print(format_output(result))
        
        # 交互式模式
        if args.interactive:
            adapter = load_adapter('srt')
            interactive_mode(adapter)
        
        # 如果没有指定操作，显示帮助
        if not (args.test or args.integration or args.command or args.interactive):
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
