#!/usr/bin/env python3
"""
Kilo Code API 模拟服务

此模块提供了一个本地模拟的Kilo Code API服务，用于在实际API不可用时进行功能验证。
模拟服务实现了与实际API相同的接口，但返回预设的响应数据。

使用方法：
    python kilocode_mock_api.py --port 8000
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional
import time
from pathlib import Path
from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kilocode_mock_api")

# 创建Flask应用
app = Flask(__name__)

# 模拟数据
MOCK_RESPONSES = {
    "health": {
        "status": "ok",
        "version": "1.0.0",
        "uptime": "12h 34m 56s"
    },
    "generate": {
        "code": """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
        """,
        "language": "python",
        "execution_time": 0.5
    },
    "interpret": {
        "description": "This function calculates the factorial of a number using recursion.",
        "complexity": {
            "time": "O(n)",
            "space": "O(n)"
        },
        "issues": [],
        "suggestions": []
    },
    "decompose": {
        "subtasks": [
            {
                "id": 1,
                "description": "Set up the project structure",
                "estimated_time": "30m"
            },
            {
                "id": 2,
                "description": "Implement the core functionality",
                "estimated_time": "2h"
            },
            {
                "id": 3,
                "description": "Write tests",
                "estimated_time": "1h"
            },
            {
                "id": 4,
                "description": "Document the code",
                "estimated_time": "30m"
            }
        ]
    },
    "optimize": {
        "optimized_code": """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
        """,
        "improvements": [
            "Replaced recursion with iteration to avoid stack overflow",
            "Reduced time complexity from O(n) to O(n) but with better constant factors",
            "Eliminated recursive function call overhead"
        ]
    },
    "analyze_complexity": {
        "time_complexity": "O(n^2)",
        "space_complexity": "O(1)",
        "details": {
            "loops": 2,
            "nested_depth": 2,
            "recursive": False
        }
    },
    "suggest_improvements": {
        "suggestions": [
            {
                "type": "performance",
                "description": "Replace bubble sort with a more efficient algorithm like quicksort",
                "impact": "high"
            },
            {
                "type": "readability",
                "description": "Add comments to explain the algorithm",
                "impact": "medium"
            },
            {
                "type": "maintainability",
                "description": "Extract the swap operation into a separate function",
                "impact": "low"
            }
        ]
    },
    "batch_generate": {
        "codes": [
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
            """
        ]
    }
}

@app.route('/v1/health', methods=['GET'])
def health():
    """健康检查接口"""
    logger.info("Health check requested")
    return jsonify(MOCK_RESPONSES["health"])

@app.route('/v1/generate', methods=['POST'])
def generate():
    """代码生成接口"""
    data = request.json
    prompt = data.get('prompt', '')
    mode = data.get('mode', 'standard')
    
    logger.info(f"Code generation requested: prompt='{prompt[:50]}...', mode='{mode}'")
    
    # 根据模式返回不同的响应
    response = MOCK_RESPONSES["generate"].copy()
    
    if mode == "explained":
        response["code"] = """
# This function calculates the factorial of a number
def factorial(n):
    # Base case: factorial of 0 or 1 is 1
    if n == 0 or n == 1:
        return 1
    # Recursive case: n! = n * (n-1)!
    else:
        return n * factorial(n-1)
        """
    elif mode == "optimized":
        response["code"] = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
        """
    
    # 模拟处理延迟
    time.sleep(0.5)
    
    return jsonify(response)

@app.route('/v1/interpret', methods=['POST'])
def interpret():
    """代码解释接口"""
    data = request.json
    code = data.get('code', '')
    
    logger.info(f"Code interpretation requested: code='{code[:50]}...'")
    
    # 模拟处理延迟
    time.sleep(0.5)
    
    return jsonify(MOCK_RESPONSES["interpret"])

@app.route('/v1/decompose', methods=['POST'])
def decompose():
    """任务分解接口"""
    data = request.json
    task = data.get('task', '')
    
    logger.info(f"Task decomposition requested: task='{task[:50]}...'")
    
    # 模拟处理延迟
    time.sleep(0.5)
    
    return jsonify(MOCK_RESPONSES["decompose"])

@app.route('/v1/optimize', methods=['POST'])
def optimize():
    """代码优化接口"""
    data = request.json
    code = data.get('code', '')
    level = data.get('level', 'medium')
    
    logger.info(f"Code optimization requested: code='{code[:50]}...', level='{level}'")
    
    # 根据优化级别返回不同的响应
    response = MOCK_RESPONSES["optimize"].copy()
    
    if level == "low":
        response["optimized_code"] = """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
        """
        response["improvements"] = ["Added base case check", "Used iteration for simple optimization"]
    elif level == "high":
        response["optimized_code"] = """
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
        """
        response["improvements"] = [
            "Added input validation",
            "Replaced recursion with iteration to avoid stack overflow",
            "Optimized loop to start from 2 instead of 1",
            "Added error handling for negative inputs"
        ]
    
    # 模拟处理延迟
    time.sleep(0.5)
    
    return jsonify(response)

@app.route('/v1/analyze_complexity', methods=['POST'])
def analyze_complexity():
    """复杂度分析接口"""
    data = request.json
    code = data.get('code', '')
    
    logger.info(f"Complexity analysis requested: code='{code[:50]}...'")
    
    # 模拟处理延迟
    time.sleep(0.5)
    
    return jsonify(MOCK_RESPONSES["analyze_complexity"])

@app.route('/v1/suggest_improvements', methods=['POST'])
def suggest_improvements():
    """代码改进建议接口"""
    data = request.json
    code = data.get('code', '')
    
    logger.info(f"Improvement suggestions requested: code='{code[:50]}...'")
    
    # 模拟处理延迟
    time.sleep(0.5)
    
    return jsonify(MOCK_RESPONSES["suggest_improvements"])

@app.route('/v1/batch_generate', methods=['POST'])
def batch_generate():
    """批量代码生成接口"""
    data = request.json
    prompts = data.get('prompts', [])
    
    logger.info(f"Batch code generation requested: {len(prompts)} prompts")
    
    # 模拟处理延迟
    time.sleep(0.5 * len(prompts))
    
    return jsonify(MOCK_RESPONSES["batch_generate"])

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Kilo Code Mock API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    args = parser.parse_args()
    
    logger.info(f"Starting Kilo Code Mock API Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
