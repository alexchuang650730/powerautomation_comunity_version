#!/usr/bin/env python3
"""
自动化测试工作流 - 使用mcpcoordinator cli模拟测试ppt agent

本模块实现了一个完整的自动化测试工作流，专注于测试ppt agent的六大特性：
1. 平台特性：PowerAutomation集成、文件格式处理、外部API集成
2. UI布局特性：专用PPT界面、进度可视化、响应式设计
3. 提示词特性：自然语言理解、模板管理、上下文提示
4. 思维特性：AI内容生成、布局优化、视觉元素建议、逻辑流程与连贯性
5. 内容特性：多模态输入处理、PPT生成引擎、多格式导出、视觉证据整合
6. 记忆特性：任务历史管理等

测试流程以mcpcoordinator cli驱动ppt agent为主线，确保所有六大特性都得到充分验证。
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ppt_agent_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ppt_agent_test")

class PptAgentTestWorkflow:
    """PPT Agent自动化测试工作流"""
    
    def __init__(self, config_path: str = "config/test_config.json"):
        """
        初始化测试工作流
        
        Args:
            config_path: 测试配置文件路径
        """
        self.config = self._load_config(config_path)
        self.results = {
            "platform_features": {},
            "ui_layout_features": {},
            "prompt_template_features": {},
            "thinking_content_generation_features": {},
            "content_features": {},
            "memory_features": {},
            "overall": {}
        }
        self.test_cases = self._prepare_test_cases()
        self.report_path = f"results/ppt_agent_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载测试配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(config_path):
                # 如果配置文件不存在，创建默认配置
                default_config = {
                    "mcpcoordinator_path": "mcptool/mcpcoordinator",
                    "test_data_dir": "cli_testing/test_data",
                    "output_dir": "results",
                    "timeout": 300,
                    "features_to_test": ["platform", "ui_layout", "prompt_template", 
                                        "thinking_content_generation", "content", "memory"],
                    "test_levels": ["unit", "integration", "e2e"]
                }
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
            
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            # 返回默认配置
            return {
                "mcpcoordinator_path": "mcptool/mcpcoordinator",
                "test_data_dir": "cli_testing/test_data",
                "output_dir": "results",
                "timeout": 300,
                "features_to_test": ["platform", "ui_layout", "prompt_template", 
                                    "thinking_content_generation", "content", "memory"],
                "test_levels": ["unit", "integration", "e2e"]
            }
    
    def _prepare_test_cases(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        准备测试用例
        
        Returns:
            按特性分类的测试用例字典
        """
        test_cases = {
            "platform": [
                {
                    "name": "测试PowerAutomation集成",
                    "command": "test_powerautomation_integration",
                    "args": ["--intent-routing", "--task-queue"],
                    "expected_result": {"status": "success"}
                },
                {
                    "name": "测试文件格式处理",
                    "command": "test_file_format_handling",
                    "args": ["--input-formats", "txt,md,csv", "--output-formats", "pptx,pdf"],
                    "expected_result": {"status": "success"}
                }
            ],
            "ui_layout": [
                {
                    "name": "测试专用PPT界面",
                    "command": "test_dedicated_ppt_interface",
                    "args": ["--template-selector", "--outline-editor"],
                    "expected_result": {"status": "success"}
                },
                {
                    "name": "测试进度可视化",
                    "command": "test_progress_visualization",
                    "args": ["--task-timeline", "--step-indicators"],
                    "expected_result": {"status": "success"}
                }
            ],
            "prompt_template": [
                {
                    "name": "测试自然语言理解",
                    "command": "test_natural_language_understanding",
                    "args": ["--intent-recognition", "--entity-extraction"],
                    "expected_result": {"status": "success"}
                },
                {
                    "name": "测试模板管理",
                    "command": "test_template_management",
                    "args": ["--builtin-templates", "--user-template-upload"],
                    "expected_result": {"status": "success"}
                }
            ],
            "thinking_content_generation": [
                {
                    "name": "测试AI内容生成",
                    "command": "test_ai_content_generation",
                    "args": ["--outline-to-slides", "--text-summarization"],
                    "expected_result": {"status": "success"}
                },
                {
                    "name": "测试布局优化",
                    "command": "test_layout_optimization",
                    "args": ["--content-aware-layout", "--visual-hierarchy"],
                    "expected_result": {"status": "success"}
                }
            ],
            "content": [
                {
                    "name": "测试多模态输入处理",
                    "command": "test_multimodal_input_handling",
                    "args": ["--text-parsing", "--data-visualization"],
                    "expected_result": {"status": "success"}
                },
                {
                    "name": "测试PPT生成引擎",
                    "command": "test_ppt_generation_engine",
                    "args": ["--engine", "python-pptx", "--master-slide-support"],
                    "expected_result": {"status": "success"}
                }
            ],
            "memory": [
                {
                    "name": "测试任务历史管理",
                    "command": "test_task_history_management",
                    "args": ["--log-level", "detailed", "--retention-policy", "permanent"],
                    "expected_result": {"status": "success"}
                }
            ]
        }
        return test_cases
    
    def run_mcpcoordinator_command(self, command: str, args: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        运行mcpcoordinator命令
        
        Args:
            command: 命令名称
            args: 命令参数
            
        Returns:
            (成功标志, 结果字典)
        """
        try:
            # 构建完整命令
            full_command = [self.config["mcpcoordinator_path"], command] + args
            logger.info(f"执行命令: {' '.join(full_command)}")
            
            # 执行命令
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=self.config["timeout"]
            )
            
            # 解析结果
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout)
                    return True, output
                except json.JSONDecodeError:
                    return True, {"raw_output": result.stdout}
            else:
                logger.error(f"命令执行失败: {result.stderr}")
                return False, {"error": result.stderr}
        except subprocess.TimeoutExpired:
            logger.error(f"命令执行超时")
            return False, {"error": "Command timed out"}
        except Exception as e:
            logger.error(f"执行命令时发生错误: {e}")
            return False, {"error": str(e)}
    
    def test_feature(self, feature: str) -> Dict[str, Any]:
        """
        测试特定特性
        
        Args:
            feature: 特性名称
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始测试特性: {feature}")
        feature_results = {}
        
        if feature not in self.test_cases:
            logger.warning(f"未找到特性 {feature} 的测试用例")
            return {"status": "skipped", "reason": "No test cases defined"}
        
        for test_case in self.test_cases[feature]:
            test_name = test_case["name"]
            logger.info(f"执行测试: {test_name}")
            
            success, result = self.run_mcpcoordinator_command(
                test_case["command"],
                test_case["args"]
            )
            
            # 验证结果
            if success:
                expected = test_case["expected_result"]
                if "status" in result and result["status"] == expected["status"]:
                    test_result = {
                        "status": "passed",
                        "details": result
                    }
                else:
                    test_result = {
                        "status": "failed",
                        "reason": "Result does not match expected",
                        "expected": expected,
                        "actual": result
                    }
            else:
                test_result = {
                    "status": "failed",
                    "reason": "Command execution failed",
                    "details": result
                }
            
            feature_results[test_name] = test_result
            logger.info(f"测试 {test_name} 结果: {test_result['status']}")
        
        # 计算特性测试的总体结果
        passed = sum(1 for r in feature_results.values() if r["status"] == "passed")
        total = len(feature_results)
        
        summary = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed/total*100:.2f}%" if total > 0 else "N/A"
        }
        
        return {
            "test_cases": feature_results,
            "summary": summary
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试
        
        Returns:
            完整测试结果
        """
        logger.info("开始运行所有测试")
        start_time = datetime.now()
        
        # 测试每个特性
        for feature in self.config["features_to_test"]:
            feature_key = f"{feature}_features"
            self.results[feature_key] = self.test_feature(feature)
        
        # 计算总体结果
        total_tests = sum(r["summary"]["total"] for r in self.results.values() if "summary" in r)
        passed_tests = sum(r["summary"]["passed"] for r in self.results.values() if "summary" in r)
        
        self.results["overall"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": f"{passed_tests/total_tests*100:.2f}%" if total_tests > 0 else "N/A",
            "duration": str(datetime.now() - start_time)
        }
        
        logger.info(f"测试完成. 总体通过率: {self.results['overall']['pass_rate']}")
        
        # 保存测试报告
        self._save_report()
        
        return self.results
    
    def _save_report(self) -> None:
        """保存测试报告"""
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"测试报告已保存至: {self.report_path}")
    
    def generate_html_report(self) -> str:
        """
        生成HTML测试报告
        
        Returns:
            HTML报告路径
        """
        html_path = self.report_path.replace('.json', '.html')
        
        # 简单的HTML报告模板
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PPT Agent 测试报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .feature {{ margin-bottom: 30px; }}
                .test-case {{ margin-left: 20px; margin-bottom: 10px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>PPT Agent 测试报告</h1>
            <div class="summary">
                <h2>总体结果</h2>
                <p>总测试数: {self.results["overall"]["total_tests"]}</p>
                <p>通过测试数: {self.results["overall"]["passed_tests"]}</p>
                <p>失败测试数: {self.results["overall"]["failed_tests"]}</p>
                <p>通过率: {self.results["overall"]["pass_rate"]}</p>
                <p>测试耗时: {self.results["overall"]["duration"]}</p>
            </div>
        """
        
        # 添加每个特性的测试结果
        for feature_key, feature_results in self.results.items():
            if feature_key == "overall" or "summary" not in feature_results:
                continue
                
            feature_name = feature_key.replace("_features", "").replace("_", " ").title()
            html_content += f"""
            <div class="feature">
                <h2>{feature_name} 特性</h2>
                <p>总测试数: {feature_results["summary"]["total"]}</p>
                <p>通过测试数: {feature_results["summary"]["passed"]}</p>
                <p>失败测试数: {feature_results["summary"]["failed"]}</p>
                <p>通过率: {feature_results["summary"]["pass_rate"]}</p>
                
                <h3>测试用例详情</h3>
                <table>
                    <tr>
                        <th>测试名称</th>
                        <th>状态</th>
                        <th>详情</th>
                    </tr>
            """
            
            for test_name, test_result in feature_results["test_cases"].items():
                status_class = "passed" if test_result["status"] == "passed" else "failed"
                details = test_result.get("reason", "") if test_result["status"] != "passed" else ""
                
                html_content += f"""
                    <tr>
                        <td>{test_name}</td>
                        <td class="{status_class}">{test_result["status"]}</td>
                        <td>{details}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML测试报告已生成: {html_path}")
        return html_path

def main():
    """主函数"""
    logger.info("启动PPT Agent自动化测试工作流")
    
    # 创建测试工作流实例
    workflow = PptAgentTestWorkflow()
    
    # 运行所有测试
    results = workflow.run_all_tests()
    
    # 生成HTML报告
    html_report = workflow.generate_html_report()
    
    logger.info(f"测试完成，总体通过率: {results['overall']['pass_rate']}")
    logger.info(f"HTML报告路径: {html_report}")
    logger.info(f"JSON报告路径: {workflow.report_path}")

if __name__ == "__main__":
    main()
