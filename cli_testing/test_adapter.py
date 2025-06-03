#!/usr/bin/env python3
"""
适配器测试工具

此工具用于测试Kilo Code和SRT适配器的功能，解决相对导入问题。
支持命令行参数和交互式模式，便于开发和测试。
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
logger = logging.getLogger("test_adapter")

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
            from adapters.srt.srt_adapter import SRTAdapter
            adapter = SRTAdapter()
            logger.info(f"Successfully loaded SRT adapter")
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
    print(f"=== Adapter Testing Interactive Mode ===")
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

def test_srt_adapter():
    """测试SRT适配器的基本功能"""
    try:
        print("\n=== Testing SRT Adapter ===")
        
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
        try:
            thought_process = "I need to solve this problem by breaking it down into steps. First, I'll analyze the requirements..."
            score = adapter.evaluate(thought_process)
            print(f"Evaluation score: {score}")
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
        
        # 测试关闭
        print("\nTesting shutdown()...")
        shutdown_result = adapter.shutdown()
        print(f"Shutdown result: {shutdown_result}")
        
        print("\nSRT Adapter tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing SRT adapter: {str(e)}")
        return False

def test_integration():
    """测试集成功能"""
    try:
        print("\n=== Testing Integration ===")
        
        # 导入集成模块
        from integration.mcptool_kilocode_integration import MCPToolKiloCodeIntegration
        from integration.rlfactory_srt_integration import RLFactorySRTIntegration
        
        # 测试MCP Tool与Kilo Code集成
        print("\nTesting MCP Tool - Kilo Code Integration...")
        kilocode_integration = MCPToolKiloCodeIntegration()
        print(f"Integration initialized: {kilocode_integration.is_initialized()}")
        
        # 测试RL Factory与SRT集成
        print("\nTesting RL Factory - SRT Integration...")
        srt_integration = RLFactorySRTIntegration()
        print(f"Integration initialized: {srt_integration.is_initialized()}")
        
        print("\nIntegration tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing integration: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Adapter Testing Tool")
    
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
                test_srt_adapter()
            
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
