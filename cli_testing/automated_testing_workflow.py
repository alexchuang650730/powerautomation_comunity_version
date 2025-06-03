#!/usr/bin/env python3
"""
自动化测试工作流实现

此模块实现了自动化测试工作流，作为第一阶段方案的两大主轴之一。
自动化测试工作流支持：
1. Manus适配的自动化测试
2. 新通用智能体的自动化测试

工作流利用多模型协同层的能力，实现端到端的自动化测试流程。
"""

import os
import sys
import json
import logging
import time
import argparse
from typing import Dict, Any, List, Optional, Union, Tuple
import threading

# 添加父目录到路径，以便导入适配器和协同层
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入多模型协同层
from integration.multi_model_synergy import MultiModelSynergy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("automated_testing_workflow")

class AutomatedTestingWorkflow:
    """
    自动化测试工作流实现，支持Manus适配的自动化测试和新通用智能体的自动化测试。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化自动化测试工作流
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._load_config(config_path)
        
        # 初始化多模型协同层
        self.synergy = MultiModelSynergy(config_path)
        
        # 初始化测试结果存储
        self.results_dir = self.config.get("results_dir", "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("Initialized automated testing workflow")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        default_config = {
            "results_dir": "../results",
            "test_types": ["manus_compatible", "new_agent"],
            "test_scenarios": ["code_generation", "code_analysis", "end_to_end"],
            "parallel_tests": 2,
            "timeout": 300
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # 合并默认配置和加载的配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            return default_config
    
    def initialize(self) -> bool:
        """
        初始化工作流
        
        Returns:
            初始化是否成功
        """
        try:
            # 初始化多模型协同层
            if not self.synergy.initialize():
                logger.error("Failed to initialize multi-model synergy layer")
                return False
            
            logger.info("Automated testing workflow initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing automated testing workflow: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """
        关闭工作流
        
        Returns:
            关闭是否成功
        """
        try:
            # 关闭多模型协同层
            if not self.synergy.shutdown():
                logger.error("Failed to shut down multi-model synergy layer")
                return False
            
            logger.info("Automated testing workflow shut down successfully")
            return True
        except Exception as e:
            logger.error(f"Error shutting down automated testing workflow: {str(e)}")
            return False
    
    def run_manus_compatible_tests(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行Manus兼容的自动化测试
        
        Args:
            test_cases: 测试用例列表
            
        Returns:
            测试结果
        """
        logger.info(f"Running {len(test_cases)} Manus compatible tests")
        
        results = {
            "test_type": "manus_compatible",
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test case {i+1}/{len(test_cases)}: {test_case.get('name', 'Unnamed test')}")
            
            try:
                # 获取测试类型和输入
                test_type = test_case.get("type", "code_generation")
                test_input = test_case.get("input", {})
                expected_output = test_case.get("expected_output", {})
                
                # 运行测试
                start_time = time.time()
                
                if test_type == "code_generation":
                    # 代码生成测试
                    prompt = test_input.get("prompt", "")
                    context = test_input.get("context", {})
                    
                    # 使用多模型协同层生成代码
                    step = {"description": prompt}
                    code_result = self.synergy.generate_code(step, context)
                    
                    # 验证结果
                    test_passed = self._validate_code_generation(code_result, expected_output)
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": test_passed,
                        "execution_time": time.time() - start_time,
                        "result": code_result
                    }
                
                elif test_type == "code_analysis":
                    # 代码分析测试
                    code = test_input.get("code", "")
                    
                    # 使用多模型协同层分析代码
                    analysis_result = self.synergy.analyze_code(code)
                    
                    # 验证结果
                    test_passed = self._validate_code_analysis(analysis_result, expected_output)
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": test_passed,
                        "execution_time": time.time() - start_time,
                        "result": analysis_result
                    }
                
                elif test_type == "end_to_end":
                    # 端到端测试
                    problem_description = test_input.get("problem_description", "")
                    
                    # 使用多模型协同层执行端到端流程
                    end_to_end_result = self.synergy.execute_end_to_end(problem_description)
                    
                    # 验证结果
                    test_passed = self._validate_end_to_end(end_to_end_result, expected_output)
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": test_passed,
                        "execution_time": time.time() - start_time,
                        "result": end_to_end_result
                    }
                
                else:
                    # 未知测试类型
                    logger.warning(f"Unknown test type: {test_type}")
                    test_passed = False
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": False,
                        "error": f"Unknown test type: {test_type}"
                    }
                
                # 更新结果统计
                if test_passed:
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
                
                # 添加测试结果
                results["test_results"].append(test_result)
                
                logger.info(f"Test case {i+1}/{len(test_cases)} {'passed' if test_passed else 'failed'} in {test_result.get('execution_time', 0):.2f}s")
                
            except Exception as e:
                # 测试执行异常
                logger.error(f"Error running test case {i+1}/{len(test_cases)}: {str(e)}")
                
                results["failed_tests"] += 1
                results["test_results"].append({
                    "name": test_case.get("name", f"Test {i+1}"),
                    "type": test_case.get("type", "unknown"),
                    "passed": False,
                    "error": str(e)
                })
        
        # 计算通过率
        results["pass_rate"] = f"{results['passed_tests'] / results['total_tests'] * 100:.2f}%" if results["total_tests"] > 0 else "N/A"
        
        logger.info(f"Manus compatible tests completed: {results['passed_tests']}/{results['total_tests']} tests passed ({results['pass_rate']})")
        
        return results
    
    def run_new_agent_tests(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行新通用智能体的自动化测试
        
        Args:
            test_cases: 测试用例列表
            
        Returns:
            测试结果
        """
        logger.info(f"Running {len(test_cases)} new agent tests")
        
        results = {
            "test_type": "new_agent",
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test case {i+1}/{len(test_cases)}: {test_case.get('name', 'Unnamed test')}")
            
            try:
                # 获取测试类型和输入
                test_type = test_case.get("type", "analysis")
                test_input = test_case.get("input", {})
                expected_output = test_case.get("expected_output", {})
                
                # 运行测试
                start_time = time.time()
                
                if test_type == "analysis":
                    # 分析能力测试
                    problem_description = test_input.get("problem_description", "")
                    
                    # 使用多模型协同层分析问题
                    analysis_result = self.synergy.analyze_problem(problem_description)
                    
                    # 验证结果
                    test_passed = self._validate_analysis(analysis_result, expected_output)
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": test_passed,
                        "execution_time": time.time() - start_time,
                        "result": analysis_result
                    }
                
                elif test_type == "planning":
                    # 规划能力测试
                    analysis = test_input.get("analysis", {})
                    
                    # 使用多模型协同层生成计划
                    plan_result = self.synergy.generate_plan(analysis)
                    
                    # 验证结果
                    test_passed = self._validate_planning(plan_result, expected_output)
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": test_passed,
                        "execution_time": time.time() - start_time,
                        "result": plan_result
                    }
                
                elif test_type == "code_generation":
                    # 代码生成能力测试
                    step = test_input.get("step", {})
                    context = test_input.get("context", {})
                    
                    # 使用多模型协同层生成代码
                    code_result = self.synergy.generate_code(step, context)
                    
                    # 验证结果
                    test_passed = self._validate_code_generation(code_result, expected_output)
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": test_passed,
                        "execution_time": time.time() - start_time,
                        "result": code_result
                    }
                
                elif test_type == "end_to_end":
                    # 一步直达能力测试
                    problem_description = test_input.get("problem_description", "")
                    
                    # 使用多模型协同层执行端到端流程
                    end_to_end_result = self.synergy.execute_end_to_end(problem_description)
                    
                    # 验证结果
                    test_passed = self._validate_end_to_end(end_to_end_result, expected_output)
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": test_passed,
                        "execution_time": time.time() - start_time,
                        "result": end_to_end_result
                    }
                
                else:
                    # 未知测试类型
                    logger.warning(f"Unknown test type: {test_type}")
                    test_passed = False
                    test_result = {
                        "name": test_case.get("name", f"Test {i+1}"),
                        "type": test_type,
                        "passed": False,
                        "error": f"Unknown test type: {test_type}"
                    }
                
                # 更新结果统计
                if test_passed:
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
                
                # 添加测试结果
                results["test_results"].append(test_result)
                
                logger.info(f"Test case {i+1}/{len(test_cases)} {'passed' if test_passed else 'failed'} in {test_result.get('execution_time', 0):.2f}s")
                
            except Exception as e:
                # 测试执行异常
                logger.error(f"Error running test case {i+1}/{len(test_cases)}: {str(e)}")
                
                results["failed_tests"] += 1
                results["test_results"].append({
                    "name": test_case.get("name", f"Test {i+1}"),
                    "type": test_case.get("type", "unknown"),
                    "passed": False,
                    "error": str(e)
                })
        
        # 计算通过率
        results["pass_rate"] = f"{results['passed_tests'] / results['total_tests'] * 100:.2f}%" if results["total_tests"] > 0 else "N/A"
        
        logger.info(f"New agent tests completed: {results['passed_tests']}/{results['total_tests']} tests passed ({results['pass_rate']})")
        
        return results
    
    def run_all_tests(self, test_cases_file: str) -> Dict[str, Any]:
        """
        运行所有测试
        
        Args:
            test_cases_file: 测试用例文件路径
            
        Returns:
            测试结果
        """
        try:
            # 加载测试用例
            with open(test_cases_file, 'r') as f:
                test_cases = json.load(f)
            
            # 分离Manus兼容测试和新通用智能体测试
            manus_tests = test_cases.get("manus_compatible_tests", [])
            new_agent_tests = test_cases.get("new_agent_tests", [])
            
            logger.info(f"Loaded {len(manus_tests)} Manus compatible tests and {len(new_agent_tests)} new agent tests")
            
            # 运行测试
            manus_results = self.run_manus_compatible_tests(manus_tests)
            new_agent_results = self.run_new_agent_tests(new_agent_tests)
            
            # 合并结果
            results = {
                "manus_compatible_results": manus_results,
                "new_agent_results": new_agent_results,
                "total_tests": manus_results["total_tests"] + new_agent_results["total_tests"],
                "passed_tests": manus_results["passed_tests"] + new_agent_results["passed_tests"],
                "failed_tests": manus_results["failed_tests"] + new_agent_results["failed_tests"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 计算总体通过率
            results["overall_pass_rate"] = f"{results['passed_tests'] / results['total_tests'] * 100:.2f}%" if results["total_tests"] > 0 else "N/A"
            
            logger.info(f"All tests completed: {results['passed_tests']}/{results['total_tests']} tests passed ({results['overall_pass_rate']})")
            
            # 保存结果
            output_file = os.path.join(self.results_dir, f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Test results saved to {output_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running all tests: {str(e)}")
            return {
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def generate_test_cases(self, requirements_file: str) -> str:
        """
        根据需求生成测试用例
        
        Args:
            requirements_file: 需求文件路径
            
        Returns:
            生成的测试用例文件路径
        """
        try:
            # 加载需求
            with open(requirements_file, 'r') as f:
                requirements = f.read()
            
            logger.info(f"Generating test cases from requirements file: {requirements_file}")
            
            # 构建提示
            prompt = f"""
Generate comprehensive test cases for the following requirements:

{requirements}

The test cases should cover both Manus compatible tests and new agent tests.
"""
            
            # 使用多模型协同层分析问题
            analysis = self.synergy.analyze_problem(prompt)
            
            # 生成测试用例
            test_cases = {
                "manus_compatible_tests": [],
                "new_agent_tests": []
            }
            
            # 为Manus兼容测试生成测试用例
            for i, subtask in enumerate(analysis.get("breakdown", [])[:5]):  # 限制测试用例数量
                description = subtask.get("description", "")
                
                if "code generation" in description.lower():
                    test_case = {
                        "name": f"Manus Code Generation Test {i+1}",
                        "type": "code_generation",
                        "input": {
                            "prompt": f"Write a function to {description.lower()}",
                            "context": {"language": "python"}
                        },
                        "expected_output": {
                            "status": "success"
                        }
                    }
                    test_cases["manus_compatible_tests"].append(test_case)
                
                elif "analysis" in description.lower():
                    test_case = {
                        "name": f"Manus Code Analysis Test {i+1}",
                        "type": "code_analysis",
                        "input": {
                            "code": "def example(n):\n    result = 0\n    for i in range(n):\n        result += i\n    return result"
                        },
                        "expected_output": {
                            "status": "success"
                        }
                    }
                    test_cases["manus_compatible_tests"].append(test_case)
                
                else:
                    test_case = {
                        "name": f"Manus End-to-End Test {i+1}",
                        "type": "end_to_end",
                        "input": {
                            "problem_description": f"Create a solution to {description.lower()}"
                        },
                        "expected_output": {
                            "status": "success"
                        }
                    }
                    test_cases["manus_compatible_tests"].append(test_case)
            
            # 为新通用智能体测试生成测试用例
            for i, subtask in enumerate(analysis.get("breakdown", [])[5:10]):  # 限制测试用例数量
                description = subtask.get("description", "")
                
                if "analysis" in description.lower():
                    test_case = {
                        "name": f"New Agent Analysis Test {i+1}",
                        "type": "analysis",
                        "input": {
                            "problem_description": f"Analyze the following problem: {description.lower()}"
                        },
                        "expected_output": {
                            "status": "success"
                        }
                    }
                    test_cases["new_agent_tests"].append(test_case)
                
                elif "plan" in description.lower():
                    test_case = {
                        "name": f"New Agent Planning Test {i+1}",
                        "type": "planning",
                        "input": {
                            "analysis": {
                                "problem": f"Create a plan for {description.lower()}",
                                "breakdown": []
                            }
                        },
                        "expected_output": {
                            "status": "success"
                        }
                    }
                    test_cases["new_agent_tests"].append(test_case)
                
                elif "code" in description.lower():
                    test_case = {
                        "name": f"New Agent Code Generation Test {i+1}",
                        "type": "code_generation",
                        "input": {
                            "step": {"description": f"Write code to {description.lower()}"},
                            "context": {"language": "python"}
                        },
                        "expected_output": {
                            "status": "success"
                        }
                    }
                    test_cases["new_agent_tests"].append(test_case)
                
                else:
                    test_case = {
                        "name": f"New Agent End-to-End Test {i+1}",
                        "type": "end_to_end",
                        "input": {
                            "problem_description": f"Create a solution to {description.lower()}"
                        },
                        "expected_output": {
                            "status": "success"
                        }
                    }
                    test_cases["new_agent_tests"].append(test_case)
            
            # 保存测试用例
            output_file = os.path.join(self.results_dir, f"test_cases_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(output_file, 'w') as f:
                json.dump(test_cases, f, indent=2)
            
            logger.info(f"Generated {len(test_cases['manus_compatible_tests'])} Manus compatible tests and {len(test_cases['new_agent_tests'])} new agent tests")
            logger.info(f"Test cases saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
            return ""
    
    def _validate_code_generation(self, result: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
        """
        验证代码生成结果
        
        Args:
            result: 代码生成结果
            expected_output: 期望输出
            
        Returns:
            验证是否通过
        """
        # 检查基本结构
        if "error" in result:
            return False
        
        if "code" not in result:
            return False
        
        # 检查代码是否为空
        if not result["code"]:
            return False
        
        # 如果期望输出中有特定检查项，则进行检查
        if "contains" in expected_output:
            for item in expected_output["contains"]:
                if item not in result["code"]:
                    return False
        
        return True
    
    def _validate_code_analysis(self, result: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
        """
        验证代码分析结果
        
        Args:
            result: 代码分析结果
            expected_output: 期望输出
            
        Returns:
            验证是否通过
        """
        # 检查基本结构
        if "error" in result:
            return False
        
        if "analysis" not in result:
            return False
        
        # 检查分析是否为空
        if not result["analysis"]:
            return False
        
        # 如果期望输出中有特定检查项，则进行检查
        if "contains" in expected_output:
            for item in expected_output["contains"]:
                if item not in str(result["analysis"]):
                    return False
        
        return True
    
    def _validate_end_to_end(self, result: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
        """
        验证端到端结果
        
        Args:
            result: 端到端结果
            expected_output: 期望输出
            
        Returns:
            验证是否通过
        """
        # 检查基本结构
        if "error" in result:
            return False
        
        if "status" not in result:
            return False
        
        # 检查状态
        if result["status"] != "success":
            return False
        
        # 检查是否包含必要的步骤结果
        required_keys = ["analysis", "plan", "code_results"]
        for key in required_keys:
            if key not in result:
                return False
        
        # 如果期望输出中有特定检查项，则进行检查
        if "contains" in expected_output:
            for item in expected_output["contains"]:
                if item not in str(result):
                    return False
        
        return True
    
    def _validate_analysis(self, result: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
        """
        验证分析结果
        
        Args:
            result: 分析结果
            expected_output: 期望输出
            
        Returns:
            验证是否通过
        """
        # 检查基本结构
        if "error" in result:
            return False
        
        if "breakdown" not in result:
            return False
        
        # 检查分解是否为空
        if not result["breakdown"]:
            return False
        
        # 如果期望输出中有特定检查项，则进行检查
        if "contains" in expected_output:
            for item in expected_output["contains"]:
                if item not in str(result):
                    return False
        
        return True
    
    def _validate_planning(self, result: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
        """
        验证规划结果
        
        Args:
            result: 规划结果
            expected_output: 期望输出
            
        Returns:
            验证是否通过
        """
        # 检查基本结构
        if "error" in result:
            return False
        
        if "steps" not in result:
            return False
        
        # 检查步骤是否为空
        if not result["steps"]:
            return False
        
        # 如果期望输出中有特定检查项，则进行检查
        if "contains" in expected_output:
            for item in expected_output["contains"]:
                if item not in str(result):
                    return False
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Automated Testing Workflow")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--requirements", type=str, help="Path to requirements file for generating test cases")
    parser.add_argument("--test_cases", type=str, help="Path to test cases file")
    parser.add_argument("--generate_only", action="store_true", help="Only generate test cases without running tests")
    parser.add_argument("--output", type=str, help="Path to output directory")
    args = parser.parse_args()
    
    try:
        # 创建自动化测试工作流
        workflow = AutomatedTestingWorkflow(args.config)
        
        # 初始化工作流
        if not workflow.initialize():
            logger.error("Failed to initialize automated testing workflow")
            sys.exit(1)
        
        try:
            # 如果指定了输出目录，则更新结果目录
            if args.output:
                workflow.results_dir = args.output
                os.makedirs(workflow.results_dir, exist_ok=True)
            
            # 如果指定了需求文件，则生成测试用例
            if args.requirements:
                test_cases_file = workflow.generate_test_cases(args.requirements)
                
                if not test_cases_file:
                    logger.error("Failed to generate test cases")
                    sys.exit(1)
                
                # 如果只生成测试用例，则退出
                if args.generate_only:
                    logger.info(f"Test cases generated successfully: {test_cases_file}")
                    sys.exit(0)
                
                # 使用生成的测试用例运行测试
                results = workflow.run_all_tests(test_cases_file)
            
            # 如果指定了测试用例文件，则运行测试
            elif args.test_cases:
                results = workflow.run_all_tests(args.test_cases)
            
            # 如果既没有指定需求文件也没有指定测试用例文件，则报错
            else:
                logger.error("Either --requirements or --test_cases must be specified")
                sys.exit(1)
            
            # 检查测试结果
            if "error" in results:
                logger.error(f"Error running tests: {results['error']}")
                sys.exit(1)
            
            # 输出测试结果摘要
            logger.info(f"Test summary: {results['passed_tests']}/{results['total_tests']} tests passed ({results['overall_pass_rate']})")
            
            # 根据通过率设置退出码
            pass_rate = results['passed_tests'] / results['total_tests'] if results['total_tests'] > 0 else 0
            if pass_rate >= 0.8:  # 80%通过率为成功
                logger.info("Tests PASSED")
                sys.exit(0)
            else:
                logger.warning("Tests FAILED")
                sys.exit(1)
                
        finally:
            # 关闭工作流
            workflow.shutdown()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
