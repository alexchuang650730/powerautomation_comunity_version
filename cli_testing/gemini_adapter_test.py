#!/usr/bin/env python3
"""
Gemini API适配器测试工具

此工具用于测试Gemini API适配器的功能，包括：
1. API健康检查
2. 代码生成功能
3. 代码解释功能
4. 任务分解功能
5. 代码优化功能
6. 复杂度分析功能

使用方法：
    python gemini_adapter_test.py --api_key YOUR_API_KEY
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
logger = logging.getLogger("gemini_adapter_test")

# 添加适配器路径
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ADAPTER_PATH not in sys.path:
    sys.path.insert(0, ADAPTER_PATH)

# 导入Gemini适配器
try:
    from adapters.kilocode.gemini_adapter import GeminiAdapter
    logger.info("Successfully imported GeminiAdapter")
except ImportError as e:
    logger.error(f"Failed to import GeminiAdapter: {str(e)}")
    sys.exit(1)

def test_health_check(adapter: GeminiAdapter) -> bool:
    """
    测试健康检查功能
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试是否通过
    """
    logger.info("Testing health check...")
    
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

def test_code_generation(adapter: GeminiAdapter) -> bool:
    """
    测试代码生成功能
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试是否通过
    """
    logger.info("Testing code generation...")
    
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
    logger.info(f"Code generation test: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def test_code_interpretation(adapter: GeminiAdapter) -> bool:
    """
    测试代码解释功能
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试是否通过
    """
    logger.info("Testing code interpretation...")
    
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
    logger.info(f"Code interpretation test: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def test_task_decomposition(adapter: GeminiAdapter) -> bool:
    """
    测试任务分解功能
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试是否通过
    """
    logger.info("Testing task decomposition...")
    
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
    logger.info(f"Task decomposition test: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def test_code_optimization(adapter: GeminiAdapter) -> bool:
    """
    测试代码优化功能
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试是否通过
    """
    logger.info("Testing code optimization...")
    
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
    logger.info(f"Code optimization test: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def test_complexity_analysis(adapter: GeminiAdapter) -> bool:
    """
    测试复杂度分析功能
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试是否通过
    """
    logger.info("Testing complexity analysis...")
    
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
    logger.info(f"Complexity analysis test: {success_count}/{len(test_cases)} tests passed ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.5  # 至少50%的测试用例通过

def test_integration(adapter: GeminiAdapter) -> bool:
    """
    测试集成功能
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试是否通过
    """
    logger.info("Testing integration functionality...")
    
    try:
        # 1. 生成代码
        prompt = "Write a Python function to find the longest common subsequence of two strings"
        logger.info(f"Generating code for: {prompt}")
        code = adapter.generate_code(prompt)
        
        if not code or len(code) == 0:
            logger.error("Failed to generate code")
            return False
        
        logger.info(f"Generated code ({len(code)} characters)")
        
        # 2. 解释代码
        logger.info("Interpreting generated code")
        interpretation = adapter.interpret_code(code)
        
        if not interpretation or not isinstance(interpretation, dict):
            logger.error("Failed to interpret code")
            return False
        
        logger.info("Code interpreted successfully")
        
        # 3. 优化代码
        logger.info("Optimizing generated code")
        optimized_code = adapter.optimize_code(code, "high")
        
        if not optimized_code or len(optimized_code) == 0:
            logger.error("Failed to optimize code")
            return False
        
        logger.info(f"Code optimized successfully ({len(optimized_code)} characters)")
        
        # 4. 分析复杂度
        logger.info("Analyzing complexity of optimized code")
        analysis = adapter.analyze_complexity(optimized_code)
        
        if not analysis or not isinstance(analysis, dict):
            logger.error("Failed to analyze code complexity")
            return False
        
        logger.info("Complexity analyzed successfully")
        
        # 5. 分解任务
        task = "Implement a system to find the longest common subsequence of multiple strings"
        logger.info(f"Decomposing task: {task}")
        subtasks = adapter.decompose_task(task)
        
        if not subtasks or not isinstance(subtasks, list) or len(subtasks) == 0:
            logger.error("Failed to decompose task")
            return False
        
        logger.info(f"Task decomposed successfully into {len(subtasks)} subtasks")
        
        logger.info("Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"Error during integration test: {str(e)}")
        return False

def run_all_tests(adapter: GeminiAdapter) -> Dict[str, bool]:
    """
    运行所有测试
    
    Args:
        adapter: Gemini适配器实例
        
    Returns:
        测试结果字典
    """
    results = {}
    
    # 测试健康检查
    results["health_check"] = test_health_check(adapter)
    
    # 测试代码生成
    results["code_generation"] = test_code_generation(adapter)
    
    # 测试代码解释
    results["code_interpretation"] = test_code_interpretation(adapter)
    
    # 测试任务分解
    results["task_decomposition"] = test_task_decomposition(adapter)
    
    # 测试代码优化
    results["code_optimization"] = test_code_optimization(adapter)
    
    # 测试复杂度分析
    results["complexity_analysis"] = test_complexity_analysis(adapter)
    
    # 测试集成功能
    results["integration"] = test_integration(adapter)
    
    return results

def save_results(results: Dict[str, bool], output_path: str) -> None:
    """
    保存测试结果
    
    Args:
        results: 测试结果字典
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
    parser = argparse.ArgumentParser(description="Gemini API Adapter Test")
    parser.add_argument("--api_key", type=str, help="Gemini API key")
    parser.add_argument("--output", type=str, default="../results/gemini_adapter_test_results.json", help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 获取API密钥
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key provided. Use --api_key or set GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    # 创建适配器
    adapter = GeminiAdapter(api_key=api_key)
    
    # 初始化适配器
    config = {
        "api_key": api_key,
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
        "max_output_tokens": 8192
    }
    
    if not adapter.initialize(config):
        logger.error("Failed to initialize adapter")
        sys.exit(1)
    
    try:
        # 运行所有测试
        results = run_all_tests(adapter)
        
        # 保存结果
        save_results(results, args.output)
        
        # 输出总结
        passed = sum(1 for result in results.values() if result is True and isinstance(result, bool))
        total = sum(1 for result in results.values() if isinstance(result, bool))
        logger.info(f"Overall test results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        
        # 根据结果设置退出码
        if passed / total >= 0.8:  # 至少80%的测试通过
            logger.info("Tests PASSED")
            sys.exit(0)
        else:
            logger.warning("Tests FAILED")
            sys.exit(1)
    finally:
        # 关闭适配器
        adapter.shutdown()

if __name__ == "__main__":
    main()
