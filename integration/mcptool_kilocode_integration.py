"""
MCP Tool与Kilo Code适配器集成模块

此模块实现了MCP Tool与Kilo Code适配器的集成，通过适配器模式最小化对原有代码的修改。
集成后，MCP Tool可以利用Kilo Code的代码生成、解释和优化能力，增强系统功能。
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

# 配置日志
logging.basicConfig(
    level=os.environ.get("MCPTOOL_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.environ.get("MCPTOOL_LOG_FILE", None)
)
logger = logging.getLogger("mcptool_kilocode_integration")

# 添加适配器路径
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'adapters'))
if ADAPTER_PATH not in sys.path:
    sys.path.append(ADAPTER_PATH)

# 导入Kilo Code适配器
try:
    from kilocode.kilocode_adapter import KiloCodeAdapter
except ImportError:
    logger.error("Failed to import KiloCodeAdapter. Please ensure the adapter is properly installed.")
    KiloCodeAdapter = None

class MCPToolKiloCodeIntegration:
    """
    MCP Tool与Kilo Code适配器的集成类
    
    此类负责初始化Kilo Code适配器，并提供与MCP Tool集成的接口。
    采用适配器模式，最小化对原有代码的修改。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化集成类
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.adapter = None
        self.config = self._load_config(config_path)
        self.initialized = False
        
        # 初始化适配器
        self._initialize_adapter()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        # 默认配置
        default_config = {
            "api_key": os.environ.get("KILO_CODE_API_KEY", ""),
            "server_url": os.environ.get("KILO_CODE_SERVER_URL", "https://api.kilocode.ai/v1"),
            "timeout": 30,
            "capabilities": {
                "code_generation": True,
                "code_interpretation": True,
                "task_decomposition": True,
                "code_optimization": True,
                "complexity_analysis": True
            }
        }
        
        # 如果没有提供配置文件路径，使用默认配置
        if not config_path:
            logger.info("Using default configuration")
            return default_config
        
        # 尝试加载配置文件
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            
            # 合并默认配置和加载的配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            logger.info("Using default configuration")
            return default_config
    
    def _initialize_adapter(self) -> bool:
        """
        初始化Kilo Code适配器
        
        Returns:
            初始化是否成功
        """
        if not KiloCodeAdapter:
            logger.error("KiloCodeAdapter is not available")
            return False
        
        try:
            # 创建适配器实例
            self.adapter = KiloCodeAdapter(
                api_key=self.config.get("api_key"),
                server_url=self.config.get("server_url")
            )
            
            # 初始化适配器
            success = self.adapter.initialize(self.config)
            
            if success:
                self.initialized = True
                logger.info("Kilo Code adapter initialized successfully")
            else:
                logger.error("Failed to initialize Kilo Code adapter")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing Kilo Code adapter: {str(e)}")
            return False
    
    def is_initialized(self) -> bool:
        """
        检查适配器是否已初始化
        
        Returns:
            适配器是否已初始化
        """
        return self.initialized and self.adapter is not None
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        获取适配器支持的能力
        
        Returns:
            支持的能力字典
        """
        if not self.is_initialized():
            logger.warning("Adapter not initialized, returning empty capabilities")
            return {}
        
        return self.adapter.get_capabilities()
    
    def generate_code(self, prompt: str, context: Optional[Dict[str, Any]] = None, 
                     mode: str = "standard") -> Dict[str, Any]:
        """
        生成代码
        
        Args:
            prompt: 代码生成提示
            context: 上下文信息
            mode: 生成模式
            
        Returns:
            包含生成代码的结果字典
        """
        if not self.is_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            code = self.adapter.generate_code(prompt, context, mode)
            return {
                "success": True,
                "code": code,
                "mode": mode
            }
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def interpret_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        解释代码
        
        Args:
            code: 需要解释的代码
            context: 上下文信息
            
        Returns:
            包含代码解释的结果字典
        """
        if not self.is_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            interpretation = self.adapter.interpret_code(code, context)
            return {
                "success": True,
                "interpretation": interpretation
            }
        except Exception as e:
            logger.error(f"Error interpreting code: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def decompose_task(self, task_description: str) -> Dict[str, Any]:
        """
        分解任务
        
        Args:
            task_description: 任务描述
            
        Returns:
            包含分解后子任务的结果字典
        """
        if not self.is_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            subtasks = self.adapter.decompose_task(task_description)
            return {
                "success": True,
                "subtasks": subtasks
            }
        except Exception as e:
            logger.error(f"Error decomposing task: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def optimize_code(self, code: str, optimization_level: str = "medium") -> Dict[str, Any]:
        """
        优化代码
        
        Args:
            code: 需要优化的代码
            optimization_level: 优化级别
            
        Returns:
            包含优化后代码的结果字典
        """
        if not self.is_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            optimized_code = self.adapter.optimize_code(code, optimization_level)
            return {
                "success": True,
                "optimized_code": optimized_code,
                "level": optimization_level
            }
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """
        分析代码复杂度
        
        Args:
            code: 需要分析的代码
            
        Returns:
            包含复杂度分析的结果字典
        """
        if not self.is_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            analysis = self.adapter.analyze_complexity(code)
            return {
                "success": True,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error analyzing code complexity: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def suggest_improvements(self, code: str) -> Dict[str, Any]:
        """
        提供代码改进建议
        
        Args:
            code: 需要分析的代码
            
        Returns:
            包含改进建议的结果字典
        """
        if not self.is_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            suggestions = self.adapter.suggest_improvements(code)
            return {
                "success": True,
                "suggestions": suggestions
            }
        except Exception as e:
            logger.error(f"Error suggesting code improvements: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def shutdown(self) -> bool:
        """
        关闭适配器，释放资源
        
        Returns:
            关闭是否成功
        """
        if not self.is_initialized():
            logger.warning("Adapter not initialized, nothing to shutdown")
            return True
        
        try:
            success = self.adapter.shutdown()
            if success:
                self.initialized = False
                self.adapter = None
                logger.info("Kilo Code adapter shut down successfully")
            else:
                logger.error("Failed to shut down Kilo Code adapter")
            
            return success
            
        except Exception as e:
            logger.error(f"Error shutting down Kilo Code adapter: {str(e)}")
            return False

# 单例实例
_instance = None

def get_instance(config_path: Optional[str] = None) -> MCPToolKiloCodeIntegration:
    """
    获取集成类的单例实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        集成类实例
    """
    global _instance
    if _instance is None:
        _instance = MCPToolKiloCodeIntegration(config_path)
    return _instance
