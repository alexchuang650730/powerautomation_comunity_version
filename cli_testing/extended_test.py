#!/usr/bin/env python3
"""
扩展测试工具 - 边界条件与异常处理测试

此工具用于测试SRT适配器和集成模块在边界条件和异常情况下的行为，
提高测试覆盖率，确保系统稳定性。
"""

import os
import sys
import json
import argparse
import logging
import importlib.util
from typing import Dict, Any, List, Optional, Union
import time
import traceback

# 添加父目录到sys.path，解决相对导入问题
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("extended_test")

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

def test_srt_edge_cases():
    """测试SRT适配器的边界条件"""
    try:
        print("\n=== Testing SRT Adapter Edge Cases ===")
        
        # 加载适配器
        adapter = load_adapter('srt')
        
        # 测试初始化 - 空配置
        print("\nTesting initialize() with empty config...")
        try:
            empty_config = {}
            init_result = adapter.initialize(empty_config)
            print(f"Initialize with empty config result: {init_result}")
        except Exception as e:
            print(f"Error in initialize with empty config: {str(e)}")
        
        # 测试初始化 - 无效设备
        print("\nTesting initialize() with invalid device...")
        try:
            invalid_device_config = {"device": "invalid_device"}
            init_result = adapter.initialize(invalid_device_config)
            print(f"Initialize with invalid device result: {init_result}")
        except Exception as e:
            print(f"Error in initialize with invalid device: {str(e)}")
        
        # 测试评估 - 空字符串
        print("\nTesting evaluate() with empty string...")
        try:
            score = adapter.evaluate("")
            print(f"Evaluation score for empty string: {score}")
        except Exception as e:
            print(f"Error in evaluate with empty string: {str(e)}")
        
        # 测试评估 - 极长字符串
        print("\nTesting evaluate() with very long string...")
        try:
            long_string = "This is a test. " * 1000
            score = adapter.evaluate(long_string)
            print(f"Evaluation score for very long string: {score}")
        except Exception as e:
            print(f"Error in evaluate with very long string: {str(e)}")
        
        # 测试训练 - 极小迭代次数
        print("\nTesting train() with minimal iterations...")
        try:
            thought_process = "Test thought process"
            train_result = adapter.train(thought_process, iterations=1)
            print(f"Training result with minimal iterations: {format_output(train_result)}")
        except Exception as e:
            print(f"Error in train with minimal iterations: {str(e)}")
        
        # 测试批量训练 - 空列表
        print("\nTesting batch_train() with empty list...")
        try:
            batch_result = adapter.batch_train([], batch_size=2)
            print(f"Batch training result with empty list: {format_output(batch_result)}")
        except Exception as e:
            print(f"Error in batch_train with empty list: {str(e)}")
        
        # 测试加载模型 - 不存在的文件
        print("\nTesting load_model() with non-existent file...")
        try:
            load_result = adapter.load_model("/tmp/non_existent_model.pt")
            print(f"Load model result with non-existent file: {load_result}")
        except Exception as e:
            print(f"Error in load_model with non-existent file: {str(e)}")
        
        # 测试关闭后操作
        print("\nTesting operations after shutdown()...")
        try:
            adapter.shutdown()
            # 尝试在关闭后评估
            score = adapter.evaluate("Test after shutdown")
            print(f"Evaluation after shutdown: {score}")
        except Exception as e:
            print(f"Error in operation after shutdown: {str(e)}")
        
        print("\nSRT Adapter edge cases tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing SRT adapter edge cases: {str(e)}")
        traceback.print_exc()
        return False

def test_integration_edge_cases():
    """测试集成模块的边界条件"""
    try:
        print("\n=== Testing Integration Edge Cases ===")
        
        # 导入集成模块
        from integration.rlfactory_srt_integration import RLFactorySRTIntegration
        
        # 测试无配置初始化
        print("\nTesting integration with no config...")
        try:
            integration = RLFactorySRTIntegration()
            print(f"Integration initialized with no config: {integration.is_initialized()}")
        except Exception as e:
            print(f"Error in integration with no config: {str(e)}")
        
        # 测试无效配置路径
        print("\nTesting integration with invalid config path...")
        try:
            integration = RLFactorySRTIntegration("/tmp/non_existent_config.json")
            print(f"Integration initialized with invalid config path: {integration.is_initialized()}")
        except Exception as e:
            print(f"Error in integration with invalid config path: {str(e)}")
        
        # 测试未初始化操作
        print("\nTesting operations before initialization...")
        try:
            # 创建新实例但不初始化
            integration = RLFactorySRTIntegration()
            integration.initialized = False
            integration.adapter = None
            
            # 尝试在未初始化状态下获取能力
            capabilities = integration.get_capabilities()
            print(f"Get capabilities before initialization: {format_output(capabilities)}")
            
            # 尝试在未初始化状态下训练
            train_result = integration.train("Test thought process")
            print(f"Train before initialization: {format_output(train_result)}")
        except Exception as e:
            print(f"Error in operations before initialization: {str(e)}")
        
        # 测试无效参数
        print("\nTesting operations with invalid parameters...")
        try:
            integration = RLFactorySRTIntegration()
            
            # 尝试使用None作为思考过程
            train_result = integration.train(None)
            print(f"Train with None thought process: {format_output(train_result)}")
            
            # 尝试使用负数迭代次数
            train_result = integration.train("Test thought process", iterations=-1)
            print(f"Train with negative iterations: {format_output(train_result)}")
            
            # 尝试使用None作为批处理大小
            batch_result = integration.batch_train(["Test1", "Test2"], batch_size=None)
            print(f"Batch train with None batch size: {format_output(batch_result)}")
        except Exception as e:
            print(f"Error in operations with invalid parameters: {str(e)}")
        
        # 测试并发操作
        print("\nTesting concurrent operations...")
        try:
            import threading
            
            integration = RLFactorySRTIntegration()
            
            def train_thread():
                try:
                    result = integration.train("Thread test", iterations=5)
                    print(f"Thread train result: {result['success']}")
                except Exception as e:
                    print(f"Thread error: {str(e)}")
            
            threads = []
            for i in range(3):
                thread = threading.Thread(target=train_thread)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
                
        except Exception as e:
            print(f"Error in concurrent operations: {str(e)}")
        
        print("\nIntegration edge cases tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing integration edge cases: {str(e)}")
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理机制"""
    try:
        print("\n=== Testing Error Handling ===")
        
        # 加载适配器
        adapter = load_adapter('srt')
        
        # 测试异常捕获和恢复
        print("\nTesting exception handling and recovery...")
        try:
            # 故意触发异常
            adapter.evaluate(123)  # 类型错误，应该是字符串
        except Exception as e:
            print(f"Expected error caught: {str(e)}")
            
            # 测试恢复能力
            try:
                score = adapter.evaluate("Recovery test after error")
                print(f"Recovery successful, evaluation score: {score}")
            except Exception as e:
                print(f"Recovery failed: {str(e)}")
        
        # 测试资源泄漏
        print("\nTesting resource leak prevention...")
        try:
            # 多次初始化和关闭，检查资源管理
            for i in range(3):
                print(f"Initialization cycle {i+1}...")
                adapter.initialize({"device": "cpu"})
                adapter.shutdown()
            
            # 最后再次初始化，确保仍然可用
            adapter.initialize({"device": "cpu"})
            score = adapter.evaluate("Resource test")
            print(f"After multiple init/shutdown cycles, evaluation score: {score}")
        except Exception as e:
            print(f"Resource management error: {str(e)}")
        
        # 测试超时处理
        print("\nTesting timeout handling...")
        try:
            import threading
            import time
            
            def timeout_function():
                try:
                    # 模拟长时间运行的操作
                    long_string = "This is a test. " * 10000
                    start_time = time.time()
                    adapter.evaluate(long_string)
                    end_time = time.time()
                    print(f"Long operation completed in {end_time - start_time:.2f} seconds")
                except Exception as e:
                    print(f"Long operation error: {str(e)}")
            
            # 启动线程
            thread = threading.Thread(target=timeout_function)
            thread.start()
            
            # 等待一段时间
            thread.join(timeout=10)
            if thread.is_alive():
                print("Operation timed out (still running after 10 seconds)")
            else:
                print("Operation completed within timeout")
        except Exception as e:
            print(f"Timeout handling error: {str(e)}")
        
        print("\nError handling tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing error handling: {str(e)}")
        traceback.print_exc()
        return False

def test_performance():
    """测试性能和稳定性"""
    try:
        print("\n=== Testing Performance and Stability ===")
        
        # 加载适配器
        adapter = load_adapter('srt')
        adapter.initialize({"device": "cpu"})
        
        # 测试批量处理性能
        print("\nTesting batch processing performance...")
        try:
            batch_sizes = [1, 2, 4, 8]
            for batch_size in batch_sizes:
                thought_processes = ["Performance test thought process"] * batch_size
                
                start_time = time.time()
                result = adapter.batch_train(thought_processes, batch_size=batch_size)
                end_time = time.time()
                
                print(f"Batch size {batch_size}: processed in {end_time - start_time:.4f} seconds")
                print(f"Average time per item: {(end_time - start_time) / batch_size:.4f} seconds")
        except Exception as e:
            print(f"Batch processing performance error: {str(e)}")
        
        # 测试连续操作稳定性
        print("\nTesting stability with continuous operations...")
        try:
            operations = 10
            print(f"Performing {operations} consecutive operations...")
            
            start_time = time.time()
            for i in range(operations):
                if i % 3 == 0:
                    adapter.evaluate(f"Stability test {i}")
                elif i % 3 == 1:
                    adapter.train(f"Stability test {i}", iterations=2)
                else:
                    adapter.improve(f"Stability test {i}")
            end_time = time.time()
            
            print(f"Completed {operations} operations in {end_time - start_time:.2f} seconds")
            print(f"Average time per operation: {(end_time - start_time) / operations:.4f} seconds")
        except Exception as e:
            print(f"Stability testing error: {str(e)}")
        
        # 测试内存使用
        print("\nTesting memory usage...")
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # 基准内存使用
            base_memory = process.memory_info().rss / 1024 / 1024
            print(f"Base memory usage: {base_memory:.2f} MB")
            
            # 执行一些操作
            for i in range(5):
                adapter.train(f"Memory test {i}", iterations=2)
            
            # 操作后内存使用
            after_memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage after operations: {after_memory:.2f} MB")
            print(f"Memory increase: {after_memory - base_memory:.2f} MB")
        except ImportError:
            print("Psutil not available, skipping memory usage test")
        except Exception as e:
            print(f"Memory usage testing error: {str(e)}")
        
        print("\nPerformance and stability tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing performance and stability: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Extended Testing Tool")
    
    parser.add_argument('--edge-cases', '-e', action='store_true',
                        help="Run edge cases tests")
    
    parser.add_argument('--integration-edge', '-i', action='store_true',
                        help="Run integration edge cases tests")
    
    parser.add_argument('--error-handling', '-r', action='store_true',
                        help="Run error handling tests")
    
    parser.add_argument('--performance', '-p', action='store_true',
                        help="Run performance and stability tests")
    
    parser.add_argument('--all', '-a', action='store_true',
                        help="Run all tests")
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 运行所有测试或指定测试
        if args.all or args.edge_cases:
            test_srt_edge_cases()
        
        if args.all or args.integration_edge:
            test_integration_edge_cases()
        
        if args.all or args.error_handling:
            test_error_handling()
        
        if args.all or args.performance:
            test_performance()
        
        # 如果没有指定测试，显示帮助
        if not (args.all or args.edge_cases or args.integration_edge or args.error_handling or args.performance):
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
