#!/usr/bin/env python3
"""
Kilo Code API 验证工具

此工具用于验证Kilo Code API的连通性和功能，包括：
1. API健康检查
2. 代码生成功能
3. 代码解释功能
4. 任务分解功能
5. 代码优化功能
6. 复杂度分析功能

使用方法：
    python kilocode_api_validator.py --config ../config/kilocode_api_config.json
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kilocode_api_validator")

# 添加适配器路径
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ADAPTER_PATH not in sys.path:
    sys.path.insert(0, ADAPTER_PATH)

# 导入Kilo Code适配器
try:
    from adapters.kilocode.kilocode_adapter import KiloCodeAdapter
    logger.info("Successfully imported KiloCodeAdapter")
except ImportError as e:
    logger.error(f"Failed to import KiloCodeAdapter: {str(e)}")
    sys.exit(1)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        return {}

def validate_health(adapter: KiloCodeAdapter) -> bool:
    """
    验证API健康状态
    
    Args:
        adapter: Kilo Code适配器实例
        
    Returns:
        健康检查是否通过
    """
    logger.info("Validating API health...")
    
    try:
        health_status = adapter.health_check()
        logger.info(f"Health check result: {health_status}")
        
        if health_status.get("status") == "ok":
            logger.info("Health check passed")
            return True
        else:
            logger.error(f"Health check failed: {health_status.get('message')}")
            return False
    except Exception as e:
        logger.error(f"Error during health check: {str(e)}")
        return False

def validate_code_generation(adapter: KiloCodeAdapter) -> bool:
    """
    验证代码生成功能
    
    Args:
        adapter: Kilo Code适配器实例
        
    Returns:
        功能验证是否通过
    """
    logger.info("Validating code generation...")
    
    test_cases = [
        {
            "prompt": "Write a Python function to calculate the factorial of a number",
            "mode": "standard"
        },
        {
            "prompt": "Create a JavaScript function to check if a string is a palindrome",
            "mode": "explained"
        },
        {
            "prompt": "Implement a binary search algorithm in C++",
            "mode": "optimized"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        try:
            logger.info(f"Test case {i+1}: {test_case['prompt'][:50]}... (mode: {test_case['mode']})")
            
            start_time = time.time()
            code = adapter.generate_code(test_case["prompt"], mode=test_case["mode"])
            elapsed_time = time.time() - start_time
            
            if code and len(code) > 0:
                logger.info(f"Generated code successfully ({len(code)} characters) in {elapsed_time:.2f} seconds")
                logger.debug(f"Generated code: {code[:100]}...")
                success_count += 1
            else:
                logger.error("Generated code is empty")
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
    
    success_rate = success_count / len(test_cases) if test_cases else 0
    logger.info(f"Code generation validation: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def validate_code_interpretation(adapter: KiloCodeAdapter) -> bool:
    """
    验证代码解释功能
    
    Args:
        adapter: Kilo Code适配器实例
        
    Returns:
        功能验证是否通过
    """
    logger.info("Validating code interpretation...")
    
    test_cases = [
        """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
        """,
        """
function isPalindrome(str) {
    const cleanStr = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    return cleanStr === cleanStr.split('').reverse().join('');
}
        """,
        """
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
    ]
    
    success_count = 0
    
    for i, code in enumerate(test_cases):
        try:
            logger.info(f"Test case {i+1}: Interpreting code...")
            
            start_time = time.time()
            interpretation = adapter.interpret_code(code)
            elapsed_time = time.time() - start_time
            
            if interpretation and isinstance(interpretation, dict) and len(interpretation) > 0:
                logger.info(f"Code interpreted successfully in {elapsed_time:.2f} seconds")
                logger.debug(f"Interpretation: {json.dumps(interpretation, indent=2)[:100]}...")
                success_count += 1
            else:
                logger.error("Code interpretation is empty or invalid")
        except Exception as e:
            logger.error(f"Error interpreting code: {str(e)}")
    
    success_rate = success_count / len(test_cases) if test_cases else 0
    logger.info(f"Code interpretation validation: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def validate_task_decomposition(adapter: KiloCodeAdapter) -> bool:
    """
    验证任务分解功能
    
    Args:
        adapter: Kilo Code适配器实例
        
    Returns:
        功能验证是否通过
    """
    logger.info("Validating task decomposition...")
    
    test_cases = [
        "Build a web scraper that extracts product information from an e-commerce website and saves it to a CSV file",
        "Create a machine learning pipeline for sentiment analysis of customer reviews",
        "Develop a RESTful API for a todo list application with user authentication"
    ]
    
    success_count = 0
    
    for i, task in enumerate(test_cases):
        try:
            logger.info(f"Test case {i+1}: {task[:50]}...")
            
            start_time = time.time()
            subtasks = adapter.decompose_task(task)
            elapsed_time = time.time() - start_time
            
            if subtasks and isinstance(subtasks, list) and len(subtasks) > 0:
                logger.info(f"Task decomposed successfully into {len(subtasks)} subtasks in {elapsed_time:.2f} seconds")
                logger.debug(f"Subtasks: {json.dumps(subtasks, indent=2)[:100]}...")
                success_count += 1
            else:
                logger.error("Task decomposition is empty or invalid")
        except Exception as e:
            logger.error(f"Error decomposing task: {str(e)}")
    
    success_rate = success_count / len(test_cases) if test_cases else 0
    logger.info(f"Task decomposition validation: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def validate_code_optimization(adapter: KiloCodeAdapter) -> bool:
    """
    验证代码优化功能
    
    Args:
        adapter: Kilo Code适配器实例
        
    Returns:
        功能验证是否通过
    """
    logger.info("Validating code optimization...")
    
    test_cases = [
        {
            "code": """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
            """,
            "level": "medium"
        },
        {
            "code": """
function bubbleSort(arr) {
    var len = arr.length;
    for (var i = 0; i < len; i++) {
        for (var j = 0; j < len - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                var temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}
            """,
            "level": "high"
        },
        {
            "code": """
for i in range(len(data)):
    for j in range(len(data)):
        if i != j:
            result += data[i] * data[j]
            """,
            "level": "low"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        try:
            logger.info(f"Test case {i+1}: Optimizing code with level {test_case['level']}...")
            
            start_time = time.time()
            optimized_code = adapter.optimize_code(test_case["code"], test_case["level"])
            elapsed_time = time.time() - start_time
            
            if optimized_code and len(optimized_code) > 0:
                logger.info(f"Code optimized successfully ({len(optimized_code)} characters) in {elapsed_time:.2f} seconds")
                logger.debug(f"Optimized code: {optimized_code[:100]}...")
                success_count += 1
            else:
                logger.error("Optimized code is empty")
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
    
    success_rate = success_count / len(test_cases) if test_cases else 0
    logger.info(f"Code optimization validation: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def validate_complexity_analysis(adapter: KiloCodeAdapter) -> bool:
    """
    验证复杂度分析功能
    
    Args:
        adapter: Kilo Code适配器实例
        
    Returns:
        功能验证是否通过
    """
    logger.info("Validating complexity analysis...")
    
    test_cases = [
        """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
        """,
        """
function quickSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }
    
    const pivot = arr[0];
    const left = [];
    const right = [];
    
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] < pivot) {
            left.push(arr[i]);
        } else {
            right.push(arr[i]);
        }
    }
    
    return [...quickSort(left), pivot, ...quickSort(right)];
}
        """,
        """
int binarySearch(int arr[], int l, int r, int x) {
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == x)
            return m;
        if (arr[m] < x)
            l = m + 1;
        else
            r = m - 1;
    }
    return -1;
}
        """
    ]
    
    success_count = 0
    
    for i, code in enumerate(test_cases):
        try:
            logger.info(f"Test case {i+1}: Analyzing code complexity...")
            
            start_time = time.time()
            analysis = adapter.analyze_complexity(code)
            elapsed_time = time.time() - start_time
            
            if analysis and isinstance(analysis, dict) and len(analysis) > 0:
                logger.info(f"Complexity analyzed successfully in {elapsed_time:.2f} seconds")
                logger.debug(f"Analysis: {json.dumps(analysis, indent=2)[:100]}...")
                success_count += 1
            else:
                logger.error("Complexity analysis is empty or invalid")
        except Exception as e:
            logger.error(f"Error analyzing complexity: {str(e)}")
    
    success_rate = success_count / len(test_cases) if test_cases else 0
    logger.info(f"Complexity analysis validation: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def validate_all(adapter: KiloCodeAdapter) -> Dict[str, bool]:
    """
    验证所有功能
    
    Args:
        adapter: Kilo Code适配器实例
        
    Returns:
        各功能验证结果字典
    """
    results = {}
    
    # 验证健康状态
    results["health"] = validate_health(adapter)
    
    # 验证代码生成
    results["code_generation"] = validate_code_generation(adapter)
    
    # 验证代码解释
    results["code_interpretation"] = validate_code_interpretation(adapter)
    
    # 验证任务分解
    results["task_decomposition"] = validate_task_decomposition(adapter)
    
    # 验证代码优化
    results["code_optimization"] = validate_code_optimization(adapter)
    
    # 验证复杂度分析
    results["complexity_analysis"] = validate_complexity_analysis(adapter)
    
    return results

def save_results(results: Dict[str, bool], output_path: str) -> None:
    """
    保存验证结果
    
    Args:
        results: 验证结果字典
        output_path: 输出文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 添加时间戳
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算总体通过率
        passed = sum(1 for result in results.values() if result is True and isinstance(result, bool))
        total = sum(1 for result in results.values() if isinstance(result, bool))
        results["overall_pass_rate"] = f"{passed}/{total} ({passed/total*100:.0f}%)" if total > 0 else "N/A"
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Kilo Code API Validator")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="../results/kilocode_api_validation_results.json", help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        sys.exit(1)
    
    # 创建适配器
    adapter = KiloCodeAdapter(
        api_key=config.get("api_key"),
        server_url=config.get("server_url")
    )
    
    # 初始化适配器
    if not adapter.initialize(config):
        logger.error("Failed to initialize adapter")
        sys.exit(1)
    
    try:
        # 验证所有功能
        results = validate_all(adapter)
        
        # 保存结果
        save_results(results, args.output)
        
        # 输出总结
        passed = sum(1 for result in results.values() if result is True and isinstance(result, bool))
        total = sum(1 for result in results.values() if isinstance(result, bool))
        logger.info(f"Overall validation: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        
        # 根据结果设置退出码
        if passed / total >= 0.8:  # 至少80%的测试通过
            logger.info("Validation PASSED")
            sys.exit(0)
        else:
            logger.warning("Validation FAILED")
            sys.exit(1)
    finally:
        # 关闭适配器
        adapter.shutdown()

if __name__ == "__main__":
    main()
