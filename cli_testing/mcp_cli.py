#!/usr/bin/env python3
"""
CLI测试工具 - MCP Coordinator模拟器

此工具用于模拟MCP Coordinator发送指令，测试Kilo Code适配器和SRT训练器的功能。
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_cli")

def load_adapter(adapter_type: str, adapter_path: str):
    """
    动态加载适配器
    
    Args:
        adapter_type: 适配器类型，'kilocode'或'srt'
        adapter_path: 适配器模块路径
        
    Returns:
        适配器实例
    """
    try:
        # 构建模块名
        if adapter_type.lower() == 'kilocode':
            module_name = "kilocode_adapter"
            class_name = "KiloCodeAdapter"
        elif adapter_type.lower() == 'srt':
            module_name = "srt_adapter"
            class_name = "SRTAdapter"
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
        
        # 构建完整路径
        full_path = os.path.join(adapter_path, f"{module_name}.py")
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Adapter file not found: {full_path}")
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取适配器类
        adapter_class = getattr(module, class_name)
        
        # 实例化适配器
        adapter = adapter_class()
        
        logger.info(f"Successfully loaded {adapter_type} adapter from {full_path}")
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
    print(f"=== MCP CLI Interactive Mode ===")
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MCP CLI - Command Line Interface for testing adapters")
    
    parser.add_argument('--adapter', '-a', required=True, choices=['kilocode', 'srt'],
                        help="Adapter type to test")
    
    parser.add_argument('--path', '-p', default='.',
                        help="Path to adapter modules (default: current directory)")
    
    parser.add_argument('--command', '-c',
                        help="Command to execute (omit for interactive mode)")
    
    parser.add_argument('--args', '-g', nargs='*', default=[],
                        help="Arguments for the command")
    
    parser.add_argument('--output', '-o',
                        help="Output file for command result (default: stdout)")
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 加载适配器
        adapter = load_adapter(args.adapter, args.path)
        
        # 检查是否有命令
        if args.command:
            # 执行单个命令
            result = execute_command(adapter, args.command, args.args)
            
            # 输出结果
            output_str = format_output(result)
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output_str)
                logger.info(f"Result written to {args.output}")
            else:
                print(output_str)
        else:
            # 进入交互式模式
            interactive_mode(adapter)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
