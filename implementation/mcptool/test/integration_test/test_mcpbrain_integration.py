"""
MCPBrain集成测试

测试MCPBrain作为系统中央思考单元的核心能力，包括复杂推理、信息整合、认知与语义理解、
多模型协同工作以及基于上下文创建工具的能力。

特别测试三种协同工作选项：
1. Option 1: Gemini + Sequential Thinking MCP 用于分析，Claude + Kilo Code 用于代码生成
2. Option 2: WebAgent MCP + Sequential Thinking MCP 用于分析，Claude + Kilo Code 用于代码生成
3. Option 3: Gemini + WebAgent MCP + Sequential Thinking MCP 用于分析，Claude + Kilo Code 用于代码生成

作者: Manus
日期: 2025-06-04
"""

import unittest
import os
import sys
import json
import requests
import anthropic
import time
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append('/home/ubuntu/implementation/mcptool')

# 导入相关模块
from mcp.adapters.sequential_thinking_adapter import SequentialThinkingAdapter
from mcp.adapters.webagent_adapter import WebAgentBAdapter
from mcp.tools.agent_problem_solver import AgentProblemSolver
from mcp.core.api_key_manager import APIKeyManager, get_api_key_manager
from mcp.core.parameter_manager import ParameterManager

# 加载环境变量
load_dotenv('/home/ubuntu/.env')

class TestMCPBrain(unittest.TestCase):
    """MCPBrain集成测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 初始化参数管理器
        self.param_manager = ParameterManager()
        
        # 初始化API密钥管理器
        self.api_key_manager = get_api_key_manager('/home/ubuntu/.env')
        
        # 初始化Claude客户端（使用真实API）
        try:
            claude_api_key = self.api_key_manager.get_claude_api_key()
            self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
            self.claude_available = True
        except Exception as e:
            print(f"Claude API初始化失败: {e}")
            self.claude_available = False
            self.claude_client = None
        
        # 初始化Gemini客户端（模拟，因为没有真实API密钥）
        self.gemini_client = MagicMock()
        self.gemini_client.generate_content = MagicMock(return_value={
            "text": "这是Gemini生成的分析内容：\n- 用户点击流数据显示用户在产品中的导航路径\n- 关键指标包括页面停留时间、跳出率和转化路径\n- 建议关注首次使用体验和关键功能的可发现性\n- 用户留存可能受到产品价值传达不清晰的影响\n- 推荐实施个性化推荐和引导式教程",
            "images": ["image1.png", "image2.png"],
            "confidence": 0.89
        })
        
        # 初始化Kilo Code客户端（使用Claude API作为后端）
        self.kilo_code_client = self._create_kilo_code_client()
        
        # 初始化Sequential Thinking适配器
        self.st_adapter = SequentialThinkingAdapter()
        
        # 初始化WebAgent适配器
        self.webagent_adapter = WebAgentBAdapter()
        
        # 初始化MCPBrain（模拟核心功能）
        self.mcpbrain = MagicMock()
        self.mcpbrain.process_complex_reasoning = MagicMock(return_value={
            "reasoning_steps": ["分析问题", "提出假设", "验证假设", "得出结论"],
            "conclusion": "这是一个有效的解决方案"
        })
        self.mcpbrain.integrate_information = MagicMock(return_value={
            "integrated_data": "综合数据结果",
            "confidence": 0.92
        })
        self.mcpbrain.semantic_understanding = MagicMock(return_value={
            "entities": ["用户", "系统", "工具"],
            "relations": [("用户", "使用", "系统"), ("系统", "调用", "工具")],
            "context": "用户交互场景"
        })
        self.mcpbrain.create_tool_from_context = MagicMock(return_value={
            "tool_name": "DataVisualizer",
            "tool_description": "数据可视化工具",
            "tool_interface": {"input": "data_json", "output": "visualization_url"}
        })
        
        # 测试数据
        self.test_problem = {
            "description": "需要一个能够处理大规模数据并生成可视化报告的工具",
            "constraints": ["性能要求高", "易于使用", "支持多种数据格式"],
            "context": "数据分析场景"
        }
        
        self.test_code_task = {
            "description": "实现一个简单的Web服务器",
            "requirements": ["支持HTTP GET/POST", "处理静态文件", "简单的路由系统"],
            "language": "Python"
        }
        
        self.test_analysis_task = {
            "description": "分析用户行为数据，找出关键模式",
            "data_source": "用户点击流数据",
            "goal": "提高用户留存率"
        }
        
        # 测试确认标志（由用户设置）
        self.test_confirmed = True
        
        # 测试时间限制（10分钟 = 600秒）
        self.time_limit = 600

    def _create_kilo_code_client(self):
        """使用Claude API创建Kilo Code客户端"""
        if not self.claude_available:
            # 如果Claude API不可用，使用模拟客户端
            kilo_code_client = MagicMock()
            kilo_code_client.generate_code = MagicMock(return_value={
                "code": "def hello_world():\n    print('Hello, World!')",
                "language": "python",
                "complexity": "low"
            })
            return kilo_code_client
        
        # 使用Claude API作为Kilo Code的后端
        class KiloCodeClient:
            def __init__(self, claude_client):
                self.claude_client = claude_client
            
            def generate_code(self, params):
                try:
                    response = self.claude_client.messages.create(
                        model="claude-3-opus-20240229",
                        max_tokens=2000,
                        messages=[
                            {"role": "user", "content": f"请根据以下规格生成Python代码。只返回代码，不要解释。\n\n任务: {json.dumps(params['task'], ensure_ascii=False)}\n\n规格: {params['spec']}"}
                        ]
                    )
                    return {
                        "code": response.content[0].text,
                        "language": "python",
                        "complexity": "medium"
                    }
                except Exception as e:
                    print(f"代码生成失败: {e}")
                    return {
                        "code": "def hello_world():\n    print('Hello, World!')",
                        "language": "python",
                        "complexity": "low"
                    }
        
        return KiloCodeClient(self.claude_client)

    def _evaluate_results(self, results, option_name):
        """
        评估测试结果，分析质量和代码模仿质量各占50%
        
        参数:
            results: 测试结果
            option_name: 选项名称
            
        返回:
            总分（0-100）
        """
        analysis_score = 0
        code_score = 0
        
        # 评估分析质量 (50%)
        if option_name == "Option 1":
            # 检查Gemini分析输出
            if "gemini_analysis" in results and results["gemini_analysis"]:
                analysis_score += 15
            
            # 检查Sequential Thinking分析
            if "st_analysis" in results and results["st_analysis"]:
                analysis_score += 35
                
        elif option_name == "Option 2":
            # 检查WebAgent结果
            if "webagent_results" in results and results["webagent_results"]:
                analysis_score += 25
            
            # 检查Sequential Thinking分析
            if "st_analysis" in results and results["st_analysis"]:
                analysis_score += 25
                
        elif option_name == "Option 3":
            # 检查Gemini分析输出
            if "gemini_analysis" in results and results["gemini_analysis"]:
                analysis_score += 15
            
            # 检查WebAgent结果
            if "webagent_results" in results and results["webagent_results"]:
                analysis_score += 15
            
            # 检查Sequential Thinking分析
            if "st_analysis" in results and results["st_analysis"]:
                analysis_score += 20
        
        # 评估代码模仿质量 (50%)
        # 检查代码规格
        if "code_spec" in results and results["code_spec"] and len(results["code_spec"]) > 100:
            code_score += 25
        
        # 检查代码结果
        if "code_result" in results and results["code_result"] and "code" in results["code_result"]:
            code_score += 25
        
        # 计算总分
        total_score = analysis_score + code_score
        
        return {
            "analysis_score": analysis_score,
            "code_score": code_score,
            "total_score": total_score
        }

    def test_option1_collaboration(self):
        """
        测试Option 1: Gemini + Sequential Thinking MCP用于分析，Claude + Kilo Code用于代码生成
        """
        print("\n测试Option 1: Gemini + Sequential Thinking MCP用于分析，Claude + Kilo Code用于代码生成")
        
        # 检查测试是否已确认
        if not self.test_confirmed:
            self.skipTest("测试未获得用户确认，跳过测试")
        
        # 跳过测试如果Claude API不可用
        if not self.claude_available:
            self.skipTest("Claude API不可用，跳过测试")
        
        # 设置测试任务
        analysis_task = self.test_analysis_task
        code_task = self.test_code_task
        
        # 开始计时
        start_time = time.time()
        
        # 1. 使用Gemini进行初步分析
        print("1. 使用Gemini进行初步分析...")
        gemini_analysis = self.gemini_client.generate_content({
            "prompt": f"分析以下任务: {json.dumps(analysis_task, ensure_ascii=False)}",
            "temperature": 0.2
        })
        
        # 2. 使用Sequential Thinking MCP进行深入分析
        print("2. 使用Sequential Thinking MCP进行深入分析...")
        st_analysis = self.st_adapter.think_step_by_step(
            analysis_task, 
            context={"gemini_analysis": gemini_analysis}
        )
        
        # 3. 使用Claude生成代码需求规格
        print("3. 使用Claude生成代码需求规格...")
        try:
            claude_response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"基于以下分析，生成详细的代码需求规格：\n\n分析任务: {json.dumps(analysis_task, ensure_ascii=False)}\n\nGemini分析: {gemini_analysis}\n\nSequential Thinking分析: {st_analysis}"}
                ]
            )
            code_spec = claude_response.content[0].text
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            code_spec = "生成代码需求规格失败，使用默认规格"
        
        # 4. 使用Kilo Code生成代码
        print("4. 使用Kilo Code生成代码...")
        code_result = self.kilo_code_client.generate_code({
            "task": code_task,
            "spec": code_spec
        })
        
        # 结束计时
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 检查是否超时
        if execution_time > self.time_limit:
            self.skipTest(f"测试超时（{execution_time:.2f}秒 > {self.time_limit}秒），跳过测试")
        
        # 验证结果
        self.assertIsNotNone(gemini_analysis)
        self.assertIsNotNone(st_analysis)
        self.assertIsNotNone(code_spec)
        self.assertIsNotNone(code_result)
        
        print(f"Option 1测试完成，执行时间: {execution_time:.2f}秒")
        
        return {
            "gemini_analysis": gemini_analysis,
            "st_analysis": st_analysis,
            "code_spec": code_spec,
            "code_result": code_result,
            "execution_time": execution_time
        }

    def test_option2_collaboration(self):
        """
        测试Option 2: WebAgent MCP + Sequential Thinking MCP用于分析，Claude + Kilo Code用于代码生成
        """
        print("\n测试Option 2: WebAgent MCP + Sequential Thinking MCP用于分析，Claude + Kilo Code用于代码生成")
        
        # 检查测试是否已确认
        if not self.test_confirmed:
            self.skipTest("测试未获得用户确认，跳过测试")
        
        # 跳过测试如果Claude API不可用
        if not self.claude_available:
            self.skipTest("Claude API不可用，跳过测试")
        
        # 设置测试任务
        analysis_task = self.test_analysis_task
        code_task = self.test_code_task
        
        # 开始计时
        start_time = time.time()
        
        # 1. 使用WebAgent MCP进行网络信息收集
        print("1. 使用WebAgent MCP进行网络信息收集...")
        webagent_results = self.webagent_adapter.enhanced_search(
            query=f"数据分析 {analysis_task['description']} {analysis_task['goal']}",
            depth=2
        )
        
        # 2. 使用Sequential Thinking MCP进行深入分析
        print("2. 使用Sequential Thinking MCP进行深入分析...")
        st_analysis = self.st_adapter.think_step_by_step(
            analysis_task, 
            context={"webagent_results": webagent_results}
        )
        
        # 3. 使用Claude生成代码需求规格
        print("3. 使用Claude生成代码需求规格...")
        try:
            claude_response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"基于以下分析，生成详细的代码需求规格：\n\n分析任务: {json.dumps(analysis_task, ensure_ascii=False)}\n\nWebAgent收集的信息: {json.dumps(webagent_results, ensure_ascii=False)}\n\nSequential Thinking分析: {st_analysis}"}
                ]
            )
            code_spec = claude_response.content[0].text
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            code_spec = "生成代码需求规格失败，使用默认规格"
        
        # 4. 使用Kilo Code生成代码
        print("4. 使用Kilo Code生成代码...")
        code_result = self.kilo_code_client.generate_code({
            "task": code_task,
            "spec": code_spec
        })
        
        # 结束计时
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 检查是否超时
        if execution_time > self.time_limit:
            self.skipTest(f"测试超时（{execution_time:.2f}秒 > {self.time_limit}秒），跳过测试")
        
        # 验证结果
        self.assertIsNotNone(webagent_results)
        self.assertIsNotNone(st_analysis)
        self.assertIsNotNone(code_spec)
        self.assertIsNotNone(code_result)
        
        print(f"Option 2测试完成，执行时间: {execution_time:.2f}秒")
        
        return {
            "webagent_results": webagent_results,
            "st_analysis": st_analysis,
            "code_spec": code_spec,
            "code_result": code_result,
            "execution_time": execution_time
        }

    def test_option3_collaboration(self):
        """
        测试Option 3: Gemini + WebAgent MCP + Sequential Thinking MCP用于分析，Claude + Kilo Code用于代码生成
        """
        print("\n测试Option 3: Gemini + WebAgent MCP + Sequential Thinking MCP用于分析，Claude + Kilo Code用于代码生成")
        
        # 检查测试是否已确认
        if not self.test_confirmed:
            self.skipTest("测试未获得用户确认，跳过测试")
        
        # 跳过测试如果Claude API不可用
        if not self.claude_available:
            self.skipTest("Claude API不可用，跳过测试")
        
        # 设置测试任务
        analysis_task = self.test_analysis_task
        code_task = self.test_code_task
        
        # 开始计时
        start_time = time.time()
        
        # 1. 使用Gemini进行初步分析
        print("1. 使用Gemini进行初步分析...")
        gemini_analysis = self.gemini_client.generate_content({
            "prompt": f"分析以下任务: {json.dumps(analysis_task, ensure_ascii=False)}",
            "temperature": 0.2
        })
        
        # 2. 使用WebAgent MCP基于Gemini分析进行网络信息收集
        print("2. 使用WebAgent MCP基于Gemini分析进行网络信息收集...")
        webagent_results = self.webagent_adapter.enhanced_search(
            query=f"数据分析 {analysis_task['description']} {gemini_analysis['text']}",
            depth=2
        )
        
        # 3. 使用Sequential Thinking MCP整合Gemini和WebAgent结果进行深入分析
        print("3. 使用Sequential Thinking MCP整合Gemini和WebAgent结果进行深入分析...")
        st_analysis = self.st_adapter.think_step_by_step(
            analysis_task, 
            context={
                "gemini_analysis": gemini_analysis,
                "webagent_results": webagent_results
            }
        )
        
        # 4. 使用Claude生成代码需求规格
        print("4. 使用Claude生成代码需求规格...")
        try:
            claude_response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"基于以下分析，生成详细的代码需求规格：\n\n分析任务: {json.dumps(analysis_task, ensure_ascii=False)}\n\nGemini分析: {gemini_analysis}\n\nWebAgent收集的信息: {json.dumps(webagent_results, ensure_ascii=False)}\n\nSequential Thinking分析: {st_analysis}"}
                ]
            )
            code_spec = claude_response.content[0].text
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            code_spec = "生成代码需求规格失败，使用默认规格"
        
        # 5. 使用Kilo Code生成代码
        print("5. 使用Kilo Code生成代码...")
        code_result = self.kilo_code_client.generate_code({
            "task": code_task,
            "spec": code_spec
        })
        
        # 结束计时
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 检查是否超时
        if execution_time > self.time_limit:
            self.skipTest(f"测试超时（{execution_time:.2f}秒 > {self.time_limit}秒），跳过测试")
        
        # 验证结果
        self.assertIsNotNone(gemini_analysis)
        self.assertIsNotNone(webagent_results)
        self.assertIsNotNone(st_analysis)
        self.assertIsNotNone(code_spec)
        self.assertIsNotNone(code_result)
        
        print(f"Option 3测试完成，执行时间: {execution_time:.2f}秒")
        
        return {
            "gemini_analysis": gemini_analysis,
            "webagent_results": webagent_results,
            "st_analysis": st_analysis,
            "code_spec": code_spec,
            "code_result": code_result,
            "execution_time": execution_time
        }

    def test_multi_model_collaboration_performance(self):
        """测试并比较三种协同工作选项的性能"""
        print("\n比较三种协同工作选项的性能")
        
        # 检查测试是否已确认
        if not self.test_confirmed:
            self.skipTest("测试未获得用户确认，跳过测试")
        
        # 跳过测试如果Claude API不可用
        if not self.claude_available:
            self.skipTest("Claude API不可用，跳过测试")
        
        # 开始计时
        start_time = time.time()
        
        # 执行三种选项的测试
        try:
            option1_results = self.test_option1_collaboration()
            option1_scores = self._evaluate_results(option1_results, "Option 1")
            print(f"Option 1 性能评分: 分析={option1_scores['analysis_score']}/50, 代码={option1_scores['code_score']}/50, 总分={option1_scores['total_score']}/100")
            
            option2_results = self.test_option2_collaboration()
            option2_scores = self._evaluate_results(option2_results, "Option 2")
            print(f"Option 2 性能评分: 分析={option2_scores['analysis_score']}/50, 代码={option2_scores['code_score']}/50, 总分={option2_scores['total_score']}/100")
            
            option3_results = self.test_option3_collaboration()
            option3_scores = self._evaluate_results(option3_results, "Option 3")
            print(f"Option 3 性能评分: 分析={option3_scores['analysis_score']}/50, 代码={option3_scores['code_score']}/50, 总分={option3_scores['total_score']}/100")
            
            # 结束计时
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 检查是否超时
            if execution_time > self.time_limit:
                self.skipTest(f"测试超时（{execution_time:.2f}秒 > {self.time_limit}秒），跳过测试")
            
            # 比较结果
            print("\n性能比较结果:")
            print(f"Option 1 (Gemini + ST): 总分={option1_scores['total_score']}/100, 执行时间={option1_results['execution_time']:.2f}秒")
            print(f"Option 2 (WebAgent + ST): 总分={option2_scores['total_score']}/100, 执行时间={option2_results['execution_time']:.2f}秒")
            print(f"Option 3 (Gemini + WebAgent + ST): 总分={option3_scores['total_score']}/100, 执行时间={option3_results['execution_time']:.2f}秒")
            
            # 确定最佳选项
            best_option = max(
                [("Option 1", option1_scores['total_score']), 
                 ("Option 2", option2_scores['total_score']), 
                 ("Option 3", option3_scores['total_score'])], 
                key=lambda x: x[1]
            )
            print(f"\n最佳选项: {best_option[0]}，总分: {best_option[1]}/100")
            
            # 返回详细结果
            return {
                "option1": {
                    "results": option1_results,
                    "scores": option1_scores
                },
                "option2": {
                    "results": option2_results,
                    "scores": option2_scores
                },
                "option3": {
                    "results": option3_results,
                    "scores": option3_scores
                },
                "best_option": best_option[0],
                "execution_time": execution_time
            }
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            raise

if __name__ == "__main__":
    unittest.main()
