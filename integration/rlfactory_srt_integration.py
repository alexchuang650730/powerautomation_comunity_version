"""
RL Factory与SRT适配器集成模块

此模块实现了RL Factory与SRT适配器的集成，通过适配器模式最小化对原有代码的修改。
集成后，RL Factory可以利用SRT的自我奖励训练机制，增强系统的学习和自进化能力。

作者: PowerAutomation团队
版本: 1.0.1
日期: 2025-06-03
"""

import os
import sys
import logging
import json
import time
import threading
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=os.environ.get("RLFACTORY_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.environ.get("RLFACTORY_LOG_FILE", None)
)
logger = logging.getLogger("rlfactory_srt_integration")

# 添加适配器路径
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ADAPTER_PATH not in sys.path:
    sys.path.insert(0, ADAPTER_PATH)  # 确保优先使用此路径

# 导入SRT适配器
try:
    from adapters.srt.srt_adapter import SRTAdapter
    logger.info("Successfully imported SRTAdapter from adapters.srt.srt_adapter")
except ImportError as e:
    logger.error(f"Failed to import SRTAdapter: {str(e)}")
    SRTAdapter = None

class RLFactorySRTIntegration:
    """
    RL Factory与SRT适配器的集成类
    
    此类负责初始化SRT适配器，并提供与RL Factory集成的接口。
    采用适配器模式，最小化对原有代码的修改。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化集成类
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 初始化状态
        self.adapter = None
        self.config = None
        self.initialized = False
        self._lock = threading.RLock()  # 用于线程安全操作
        
        # 加载配置
        self.config = self._load_config(config_path)
        
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
            "model_path": os.environ.get("SRT_MODEL_PATH", ""),
            "batch_size": 32,
            "learning_rate": 1e-5,
            "max_iterations": 1000,
            "evaluation_interval": 100,
            "save_interval": 500,
            "device": "cuda" if os.environ.get("SRT_USE_CUDA", "1") == "1" else "cpu",
            "capabilities": {
                "self_reward_training": True,
                "thought_process_evaluation": True,
                "thought_process_improvement": True
            },
            "retry": {
                "max_attempts": 3,
                "delay_seconds": 1,
                "backoff_factor": 2
            }
        }
        
        # 如果没有提供配置文件路径，使用默认配置
        if not config_path:
            logger.info("Using default configuration")
            return default_config
        
        # 尝试加载配置文件
        try:
            # 确保文件存在
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found: {config_path}, using default configuration")
                return default_config
            
            # 读取配置文件
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            
            # 合并默认配置和加载的配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config[key], dict):
                    # 递归合并嵌套字典
                    for nested_key, nested_value in value.items():
                        if nested_key not in config[key]:
                            config[key][nested_key] = nested_value
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            logger.info("Using default configuration")
            return default_config
    
    def _initialize_adapter(self) -> bool:
        """
        初始化SRT适配器
        
        使用线程安全的方式初始化适配器，支持重试机制。
        
        Returns:
            初始化是否成功
        """
        # 线程安全操作
        with self._lock:
            # 检查适配器是否已初始化
            if self.initialized and self.adapter is not None:
                logger.info("SRT adapter already initialized")
                return True
            
            # 检查SRT适配器是否可用
            if not SRTAdapter:
                logger.error("SRTAdapter is not available")
                return False
            
            # 获取重试配置
            retry_config = self.config.get("retry", {})
            max_attempts = retry_config.get("max_attempts", 3)
            delay_seconds = retry_config.get("delay_seconds", 1)
            backoff_factor = retry_config.get("backoff_factor", 2)
            
            # 尝试初始化适配器，支持重试
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                try:
                    # 创建适配器实例
                    self.adapter = SRTAdapter(
                        model_path=self.config.get("model_path", ""),
                        device=self.config.get("device", "cpu")
                    )
                    logger.info("SRTAdapter instance created successfully")
                    
                    # 初始化适配器
                    success = self.adapter.initialize(self.config)
                    
                    if success:
                        self.initialized = True
                        logger.info("SRT adapter initialized successfully")
                        return True
                    else:
                        logger.error(f"Failed to initialize SRT adapter (attempt {attempt}/{max_attempts})")
                        
                        # 如果是最后一次尝试，放弃
                        if attempt >= max_attempts:
                            break
                        
                        # 等待后重试
                        wait_time = delay_seconds * (backoff_factor ** (attempt - 1))
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        
                except Exception as e:
                    logger.error(f"Error initializing SRT adapter (attempt {attempt}/{max_attempts}): {str(e)}")
                    
                    # 如果是最后一次尝试，放弃
                    if attempt >= max_attempts:
                        break
                    
                    # 等待后重试
                    wait_time = delay_seconds * (backoff_factor ** (attempt - 1))
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            # 所有尝试都失败
            logger.error(f"Failed to initialize SRT adapter after {max_attempts} attempts")
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
        
        try:
            return self.adapter.get_capabilities()
        except Exception as e:
            logger.error(f"Error getting capabilities: {str(e)}")
            return {}
    
    def _ensure_initialized(self) -> bool:
        """
        确保适配器已初始化
        
        如果适配器未初始化，尝试初始化它。
        
        Returns:
            适配器是否已初始化
        """
        if self.is_initialized():
            return True
        
        logger.warning("Adapter not initialized, attempting to initialize")
        return self._initialize_adapter()
    
    def train(self, thought_process: Union[str, Dict[str, Any]], 
             iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        使用自我奖励机制训练模型
        
        Args:
            thought_process: 思考过程
            iterations: 训练迭代次数，如果为None则使用配置中的值
            
        Returns:
            包含训练结果的字典
        """
        # 确保适配器已初始化
        if not self._ensure_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        # 验证参数
        if thought_process is None:
            logger.error("Thought process cannot be None")
            return {"success": False, "error": "Thought process cannot be None"}
        
        if iterations is not None and iterations <= 0:
            logger.error(f"Invalid iterations value: {iterations}")
            return {"success": False, "error": f"Invalid iterations value: {iterations}"}
        
        try:
            # 如果未指定迭代次数，使用配置中的值
            if iterations is None:
                iterations = self.config.get("max_iterations", 1000)
            
            # 线程安全操作
            with self._lock:
                result = self.adapter.train(thought_process, iterations)
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def batch_train(self, thought_processes: List[Union[str, Dict[str, Any]]], 
                  batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        批量训练模型
        
        Args:
            thought_processes: 思考过程列表
            batch_size: 批处理大小，如果为None则使用配置中的值
            
        Returns:
            包含训练结果的字典
        """
        # 确保适配器已初始化
        if not self._ensure_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        # 验证参数
        if not thought_processes:
            logger.warning("Empty thought processes list")
            return {
                "success": True,
                "result": {
                    "batches": 0,
                    "samples": 0,
                    "batch_results": [],
                    "average_reward": 0.0
                }
            }
        
        try:
            # 如果未指定批处理大小，使用配置中的值
            if batch_size is None:
                batch_size = self.config.get("batch_size", 32)
            
            # 线程安全操作
            with self._lock:
                result = self.adapter.batch_train(thought_processes, batch_size)
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error batch training model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def evaluate(self, thought_process: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估思考过程的质量
        
        Args:
            thought_process: 需要评估的思考过程
            
        Returns:
            包含评估结果的字典
        """
        # 确保适配器已初始化
        if not self._ensure_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        # 验证参数
        if thought_process is None:
            logger.error("Thought process cannot be None")
            return {"success": False, "error": "Thought process cannot be None"}
        
        try:
            # 线程安全操作
            with self._lock:
                score = self.adapter.evaluate(thought_process)
            
            return {
                "success": True,
                "score": score
            }
        except Exception as e:
            logger.error(f"Error evaluating thought process: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def improve(self, thought_process: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        改进思考过程
        
        Args:
            thought_process: 原始思考过程
            
        Returns:
            包含改进后思考过程的字典
        """
        # 确保适配器已初始化
        if not self._ensure_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        # 验证参数
        if thought_process is None:
            logger.error("Thought process cannot be None")
            return {"success": False, "error": "Thought process cannot be None"}
        
        try:
            # 线程安全操作
            with self._lock:
                improved = self.adapter.improve(thought_process)
            
            return {
                "success": True,
                "improved_thought_process": improved
            }
        except Exception as e:
            logger.error(f"Error improving thought process: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_model(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        保存模型
        
        Args:
            path: 模型保存路径，如果为None则使用配置中的路径
            
        Returns:
            包含保存结果的字典
        """
        # 确保适配器已初始化
        if not self._ensure_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            # 如果未指定路径，使用配置中的路径
            if path is None:
                path = self.config.get("model_path", "")
            
            # 如果路径为空，生成默认路径
            if not path:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join("models", f"srt_model_{timestamp}.pt")
                
                # 确保目录存在
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 线程安全操作
            with self._lock:
                success = self.adapter.save_model(path)
            
            return {
                "success": success,
                "path": path
            }
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_model(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载模型
        
        Args:
            path: 模型加载路径，如果为None则使用配置中的路径
            
        Returns:
            包含加载结果的字典
        """
        # 确保适配器已初始化
        if not self._ensure_initialized():
            return {"success": False, "error": "Adapter not initialized"}
        
        try:
            # 如果未指定路径，使用配置中的路径
            if path is None:
                path = self.config.get("model_path", "")
            
            # 如果路径为空，返回错误
            if not path:
                return {
                    "success": False,
                    "error": "Model path not specified"
                }
            
            # 检查文件是否存在
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"Model file not found: {path}"
                }
            
            # 线程安全操作
            with self._lock:
                success = self.adapter.load_model(path)
            
            return {
                "success": success,
                "path": path
            }
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
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
        # 线程安全操作
        with self._lock:
            if not self.is_initialized():
                logger.warning("Adapter not initialized, nothing to shutdown")
                return True
            
            try:
                success = self.adapter.shutdown()
                if success:
                    self.initialized = False
                    self.adapter = None
                    logger.info("SRT adapter shut down successfully")
                else:
                    logger.error("Failed to shut down SRT adapter")
                
                return success
                
            except Exception as e:
                logger.error(f"Error shutting down SRT adapter: {str(e)}")
                return False

# 单例实例
_instance = None
_instance_lock = threading.Lock()

def get_instance(config_path: Optional[str] = None) -> RLFactorySRTIntegration:
    """
    获取集成类的单例实例
    
    线程安全的单例模式实现。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        集成类实例
    """
    global _instance
    
    # 检查实例是否已存在
    if _instance is not None:
        return _instance
    
    # 线程安全地创建实例
    with _instance_lock:
        # 再次检查，防止在获取锁的过程中其他线程已创建实例
        if _instance is None:
            _instance = RLFactorySRTIntegration(config_path)
    
    return _instance

def reset_instance() -> None:
    """
    重置单例实例
    
    用于测试或重新初始化。
    """
    global _instance
    
    with _instance_lock:
        if _instance is not None:
            _instance.shutdown()
            _instance = None
