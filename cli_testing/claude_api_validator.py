#!/usr/bin/env python3
"""
Claude API验证工具

此工具用于验证Claude API密钥的有效性，并测试基本功能。
使用方法：
    python claude_api_validator.py --api_key YOUR_API_KEY
"""

import os
import sys
import json
import argparse
import logging
import time
import requests
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude_api_validator")

class ClaudeAPIValidator:
    """Claude API验证类"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        """
        初始化验证器
        
        Args:
            api_key: Claude API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = "claude-3-opus-20240229"  # 默认使用最新模型
        
        # 初始化会话
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        })
        
        logger.info(f"Initialized Claude API validator with base URL: {self.base_url}")
    
    def validate_api_key(self) -> Dict[str, Any]:
        """
        验证API密钥有效性
        
        Returns:
            验证结果字典
        """
        try:
            # 构建简单的请求
            payload = {
                "model": self.model,
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello, are you working? Please respond with a simple confirmation."}
                ]
            }
            
            logger.info("Sending test request to Claude API...")
            
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("content", [{}])[0].get("text", "")
                
                return {
                    "status": "success",
                    "message": "API key is valid",
                    "model": self.model,
                    "response": content[:100] + "..." if len(content) > 100 else content
                }
            else:
                error_message = f"API request failed with status code {response.status_code}"
                try:
                    error_details = response.json()
                    error_message += f": {error_details.get('error', {}).get('message', '')}"
                except:
                    error_message += f": {response.text}"
                
                return {
                    "status": "error",
                    "message": error_message,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception occurred: {str(e)}",
                "exception": str(e)
            }
    
    def test_code_generation(self) -> Dict[str, Any]:
        """
        测试代码生成功能
        
        Returns:
            测试结果字典
        """
        try:
            # 构建代码生成请求
            payload = {
                "model": self.model,
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": "Write a Python function to calculate the factorial of a number using recursion."}
                ]
            }
            
            logger.info("Testing code generation...")
            
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("content", [{}])[0].get("text", "")
                
                # 检查是否包含Python代码
                if "def factorial" in content and "return" in content:
                    return {
                        "status": "success",
                        "message": "Code generation test passed",
                        "code": content
                    }
                else:
                    return {
                        "status": "warning",
                        "message": "Response received but may not contain proper code",
                        "response": content[:100] + "..." if len(content) > 100 else content
                    }
            else:
                return {
                    "status": "error",
                    "message": f"API request failed with status code {response.status_code}",
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception occurred: {str(e)}",
                "exception": str(e)
            }
    
    def test_code_explanation(self) -> Dict[str, Any]:
        """
        测试代码解释功能
        
        Returns:
            测试结果字典
        """
        try:
            # 构建代码解释请求
            code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
            
            payload = {
                "model": self.model,
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": f"Explain the following code and analyze its time complexity:\n\n```python\n{code}\n```"}
                ]
            }
            
            logger.info("Testing code explanation...")
            
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("content", [{}])[0].get("text", "")
                
                # 检查是否包含解释和复杂度分析
                if "quicksort" in content and ("complexity" in content.lower() or "O(n log n)" in content):
                    return {
                        "status": "success",
                        "message": "Code explanation test passed",
                        "explanation": content[:200] + "..." if len(content) > 200 else content
                    }
                else:
                    return {
                        "status": "warning",
                        "message": "Response received but may not contain proper explanation",
                        "response": content[:100] + "..." if len(content) > 100 else content
                    }
            else:
                return {
                    "status": "error",
                    "message": f"API request failed with status code {response.status_code}",
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception occurred: {str(e)}",
                "exception": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试
        
        Returns:
            所有测试结果
        """
        results = {}
        
        # 验证API密钥
        logger.info("Validating API key...")
        results["api_key_validation"] = self.validate_api_key()
        
        # 如果API密钥有效，继续其他测试
        if results["api_key_validation"]["status"] == "success":
            logger.info("API key is valid, running additional tests...")
            
            # 测试代码生成
            results["code_generation"] = self.test_code_generation()
            
            # 测试代码解释
            results["code_explanation"] = self.test_code_explanation()
            
            # 计算总体结果
            success_count = sum(1 for test in results.values() if test["status"] == "success")
            total_count = len(results)
            results["summary"] = {
                "success_count": success_count,
                "total_count": total_count,
                "success_rate": f"{success_count / total_count * 100:.1f}%",
                "overall_status": "success" if success_count == total_count else "partial" if success_count > 0 else "failure"
            }
        else:
            logger.error("API key validation failed, skipping additional tests")
            results["summary"] = {
                "success_count": 0,
                "total_count": 1,
                "success_rate": "0.0%",
                "overall_status": "failure"
            }
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Claude API Validator")
    parser.add_argument("--api_key", type=str, help="Claude API key")
    parser.add_argument("--output", type=str, default="../results/claude_api_validation_results.json", help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 获取API密钥
    api_key = args.api_key or os.environ.get("CLAUDE_API_KEY")
    
    if not api_key:
        logger.error("No API key provided. Use --api_key or set CLAUDE_API_KEY environment variable.")
        sys.exit(1)
    
    # 创建验证器
    validator = ClaudeAPIValidator(api_key)
    
    # 运行测试
    results = validator.run_all_tests()
    
    # 输出结果
    logger.info(f"Test summary: {results['summary']['success_count']}/{results['summary']['total_count']} tests passed ({results['summary']['success_rate']})")
    logger.info(f"Overall status: {results['summary']['overall_status']}")
    
    # 保存结果
    if args.output:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # 添加时间戳
            results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 保存结果
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    # 设置退出码
    if results["summary"]["overall_status"] == "success":
        sys.exit(0)
    elif results["summary"]["overall_status"] == "partial":
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()
