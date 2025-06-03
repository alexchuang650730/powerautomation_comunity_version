#!/usr/bin/env python3
"""
PowerAutomation 三层架构图生成脚本
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图形
plt.figure(figsize=(16, 12))

# 定义颜色
LAYER_COLORS = {
    "application": "#1a5276",  # 深蓝色
    "enhancement": "#2874a6",  # 中蓝色
    "foundation": "#3498db",   # 浅蓝色
    "component": "#aed6f1",    # 最浅蓝色
    "arrow": "#154360",        # 箭头颜色
    "text": "#ffffff",         # 文字颜色
    "background": "#f5f5f5"    # 背景色
}

# 设置背景色
ax = plt.gca()
ax.set_facecolor(LAYER_COLORS["background"])

# 定义层高度
layer_heights = {
    "application": 0.8,
    "enhancement": 0.5,
    "foundation": 0.2
}

# 定义层宽度
layer_width = 0.9

# 绘制层背景
def draw_layer(name, y_pos, height, width=layer_width, color=None):
    if color is None:
        color = LAYER_COLORS.get(name.lower(), LAYER_COLORS["component"])
    
    rect = patches.Rectangle(
        (0.05, y_pos), 
        width, 
        height, 
        linewidth=2, 
        edgecolor='black', 
        facecolor=color, 
        alpha=0.3,
        zorder=1
    )
    ax.add_patch(rect)
    
    # 添加层名称
    plt.text(
        0.07, 
        y_pos + height/2, 
        name, 
        fontsize=18, 
        fontweight='bold', 
        va='center',
        zorder=3
    )

# 绘制组件
def draw_component(name, x_pos, y_pos, width=0.15, height=0.1, color=None, zorder=2):
    if color is None:
        color = LAYER_COLORS["component"]
    
    rect = patches.Rectangle(
        (x_pos, y_pos), 
        width, 
        height, 
        linewidth=1.5, 
        edgecolor='black', 
        facecolor=color, 
        alpha=0.9,
        zorder=zorder
    )
    ax.add_patch(rect)
    
    # 添加组件名称
    plt.text(
        x_pos + width/2, 
        y_pos + height/2, 
        name, 
        fontsize=12, 
        fontweight='bold', 
        ha='center', 
        va='center',
        color=LAYER_COLORS["text"],
        zorder=zorder+1
    )
    
    return (x_pos + width/2, y_pos + height/2)  # 返回中心点坐标

# 绘制子组件
def draw_subcomponent(name, parent_x, parent_y, width=0.13, height=0.04, offset_x=0, offset_y=-0.03, color=None, zorder=3):
    if color is None:
        color = LAYER_COLORS["component"]
    
    x_pos = parent_x - width/2 + offset_x
    y_pos = parent_y - height/2 + offset_y
    
    rect = patches.Rectangle(
        (x_pos, y_pos), 
        width, 
        height, 
        linewidth=1, 
        edgecolor='black', 
        facecolor='white', 
        alpha=0.9,
        zorder=zorder
    )
    ax.add_patch(rect)
    
    # 添加子组件名称
    plt.text(
        x_pos + width/2, 
        y_pos + height/2, 
        name, 
        fontsize=9, 
        ha='center', 
        va='center',
        zorder=zorder+1
    )

# 绘制连接线
def draw_arrow(start, end, color=None, style='-', width=1.5, zorder=1, bidirectional=False):
    if color is None:
        color = LAYER_COLORS["arrow"]
    
    # 提取坐标
    start_x, start_y = start
    end_x, end_y = end
    
    # 计算箭头方向
    dx = end_x - start_x
    dy = end_y - start_y
    
    # 绘制箭头
    if bidirectional:
        plt.arrow(start_x, start_y, dx, dy, head_width=0.02, head_length=0.02, 
                 fc=color, ec=color, linewidth=width, length_includes_head=True,
                 linestyle=style, zorder=zorder)
        plt.arrow(end_x, end_y, -dx, -dy, head_width=0.02, head_length=0.02, 
                 fc=color, ec=color, linewidth=width, length_includes_head=True,
                 linestyle=style, zorder=zorder)
    else:
        plt.arrow(start_x, start_y, dx, dy, head_width=0.02, head_length=0.02, 
                 fc=color, ec=color, linewidth=width, length_includes_head=True,
                 linestyle=style, zorder=zorder)

# 绘制三层架构
draw_layer("应用层 (APPLICATION LAYER)", layer_heights["foundation"] + layer_heights["enhancement"] + 0.05, layer_heights["application"], color=LAYER_COLORS["application"])
draw_layer("增强层 (ENHANCEMENT LAYER)", layer_heights["foundation"] + 0.05, layer_heights["enhancement"], color=LAYER_COLORS["enhancement"])
draw_layer("基础层 (FOUNDATION LAYER)", 0.05, layer_heights["foundation"], color=LAYER_COLORS["foundation"])

# 绘制应用层组件
agent_solver = draw_component("Agent\nProblem Solver", 0.2, 0.85, color=LAYER_COLORS["application"])
proactive_solver = draw_component("Proactive\nProblem Solver", 0.4, 0.85, color=LAYER_COLORS["application"])
release_manager = draw_component("Release\nManager", 0.6, 0.85, color=LAYER_COLORS["application"])
thought_recorder = draw_component("Thought Action\nRecorder", 0.8, 0.85, color=LAYER_COLORS["application"])
supermemory = draw_component("Supermemory\nIntegration", 1.0, 0.85, color=LAYER_COLORS["application"])

# 绘制增强层组件
rl_factory = draw_component("RL Factory", 0.2, 0.6, color=LAYER_COLORS["enhancement"])
srt_integration = draw_component("S&T Integration", 0.5, 0.6, color=LAYER_COLORS["enhancement"])
evoagent = draw_component("EveAgentX\nAlgorithms", 0.8, 0.6, color=LAYER_COLORS["enhancement"])
evolution = draw_component("Evolution\nAlgorithms", 1.0, 0.6, color=LAYER_COLORS["enhancement"])

# 绘制基础层组件
mcp_tool = draw_component("MCP Tool", 0.3, 0.25, width=0.2, color=LAYER_COLORS["foundation"])
kilo_code = draw_component("Kilo Code Integration", 0.8, 0.25, width=0.25, color=LAYER_COLORS["foundation"])

# 绘制子组件
# RL Factory 子组件
draw_subcomponent("Iteration AutoELO", 0.2, 0.6, offset_y=-0.03)
draw_subcomponent("Riccati Learner", 0.2, 0.6, offset_y=-0.08)

# S&T Integration 子组件
draw_subcomponent("Dense Area Attention", 0.5, 0.6, offset_y=-0.03)
draw_subcomponent("BLGO Distribution", 0.5, 0.6, offset_y=-0.08)

# Evolution Algorithms 子组件
draw_subcomponent("Actions Benchmarks", 1.0, 0.6, offset_y=-0.03)

# MCP Tool 子组件
draw_subcomponent("API Mediator", 0.3, 0.25, offset_y=-0.03)
draw_subcomponent("G% Code Quality", 0.3, 0.25, offset_y=-0.08)
draw_subcomponent("Centaboo's", 0.3, 0.25, offset_y=-0.13)

# Kilo Code Integration 子组件
draw_subcomponent("Code Generation", 0.8, 0.25, offset_y=-0.03)
draw_subcomponent("Integration Aligners", 0.8, 0.25, offset_y=-0.08)

# 绘制连接线 - 应用层到增强层
draw_arrow(agent_solver, rl_factory)
draw_arrow(proactive_solver, srt_integration)
draw_arrow(release_manager, srt_integration)
draw_arrow(thought_recorder, evoagent)
draw_arrow(supermemory, evolution)

# 绘制连接线 - 增强层内部
draw_arrow((0.25, 0.6), (0.45, 0.6))  # RL Factory 到 S&T
draw_arrow((0.55, 0.6), (0.75, 0.6))  # S&T 到 EvoAgentX
draw_arrow((0.85, 0.6), (0.95, 0.6))  # EvoAgentX 到 Evolution

# 绘制连接线 - 增强层到基础层
draw_arrow(rl_factory, (0.3, 0.35))  # RL Factory 到 MCP Tool
draw_arrow(srt_integration, (0.3, 0.35))  # S&T 到 MCP Tool
draw_arrow(evoagent, (0.8, 0.35))  # EvoAgentX 到 Kilo Code
draw_arrow(evolution, (0.8, 0.35))  # Evolution 到 Kilo Code

# 绘制连接线 - 基础层内部
draw_arrow((0.4, 0.25), (0.7, 0.25), bidirectional=True)  # MCP Tool 和 Kilo Code 双向

# 添加标题
plt.text(0.5, 0.97, "POWERAUTOMATION 三层架构", fontsize=24, fontweight='bold', ha='center')

# 设置坐标轴
plt.xlim(0, 1.2)
plt.ylim(0, 1)
plt.axis('off')

# 保存图像
plt.savefig('/home/ubuntu/powerautomation_integration/phase1/images/powerautomation_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/ubuntu/powerautomation_integration/phase1/images/powerautomation_architecture.svg', format='svg', bbox_inches='tight')

print("架构图已生成并保存到 /home/ubuntu/powerautomation_integration/phase1/images/ 目录")
