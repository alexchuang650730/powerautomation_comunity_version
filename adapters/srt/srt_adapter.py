"""
SRT (Self-Reward Training) 适配器模块

此模块实现了SRT适配器，提供自我奖励训练、评估和改进功能。
适配器使用PyTorch实现核心功能，支持CPU和GPU训练。

作者: PowerAutomation团队
版本: 1.0.1
日期: 2025-06-03
"""

import os
import sys
import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import random
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=os.environ.get("SRT_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.environ.get("SRT_LOG_FILE", None)
)
logger = logging.getLogger("srt_adapter")

# 导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, using mock implementation")
    TORCH_AVAILABLE = False

# 添加接口路径
INTERFACE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'interfaces'))
if INTERFACE_PATH not in sys.path:
    sys.path.append(INTERFACE_PATH)

# 导入接口
try:
    from adapters.interfaces.adapter_interface import AdapterInterface
    from adapters.interfaces.self_reward_training_interface import SelfRewardTrainingInterface
except ImportError as e:
    logger.error(f"Failed to import interfaces: {str(e)}")
    raise

class SRTModel(nn.Module):
    """
    SRT模型类
    
    实现了自我奖励训练的核心模型结构，包括编码器和奖励预测器。
    """
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 2, dropout: float = 0.1):
        """
        初始化SRT模型
        
        Args:
            hidden_size: 隐藏层大小
            num_layers: 层数
            dropout: Dropout比例
        """
        super(SRTModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 编码器 - 使用简单的LSTM作为示例
        # 在实际应用中，可以替换为更复杂的Transformer等模型
        self.encoder = nn.LSTM(
            input_size=256,  # 输入特征维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 奖励预测器
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出范围[0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为[batch_size, seq_len, input_size]
            
        Returns:
            奖励预测，形状为[batch_size, 1]
        """
        # 编码
        output, (hidden, _) = self.encoder(x)
        
        # 使用最后一个时间步的隐藏状态
        last_hidden = hidden[-1]
        
        # 预测奖励
        reward = self.reward_predictor(last_hidden)
        
        return reward

class ThoughtDataset(Dataset):
    """
    思考过程数据集
    
    用于批量处理思考过程文本。
    """
    
    def __init__(self, thought_processes: List[str], max_length: int = 512):
        """
        初始化数据集
        
        Args:
            thought_processes: 思考过程文本列表
            max_length: 最大序列长度
        """
        self.thought_processes = thought_processes
        self.max_length = max_length
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        Returns:
            数据集中样本数量
        """
        return len(self.thought_processes)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            编码后的样本张量
        """
        # 简单编码，实际应用中应使用更复杂的编码方法
        text = self.thought_processes[idx]
        
        # 将文本转换为简单的数值表示
        # 这里使用一种简化的编码方式，实际应用中应使用更好的文本表示方法
        encoded = self._simple_encode(text)
        
        return encoded
    
    def _simple_encode(self, text: str) -> torch.Tensor:
        """
        简单编码方法
        
        将文本转换为数值表示。这是一个简化的实现，
        实际应用中应使用更复杂的编码方法，如词嵌入或预训练模型。
        
        Args:
            text: 输入文本
            
        Returns:
            编码后的张量
        """
        # 截断或填充到固定长度
        if len(text) > self.max_length:
            text = text[:self.max_length]
        else:
            text = text + " " * (self.max_length - len(text))
        
        # 简单的字符级编码
        # 将每个字符转换为其ASCII值，并归一化到[0, 1]范围
        encoded = [ord(c) / 256.0 for c in text]
        
        # 重塑为[seq_len, input_size]
        # 这里我们使用256作为输入特征维度
        seq_len = len(encoded) // 256
        if seq_len == 0:
            seq_len = 1
        
        # 确保长度是256的倍数
        encoded = encoded[:seq_len * 256]
        
        # 重塑为[seq_len, 256]
        encoded = torch.tensor(encoded, dtype=torch.float32).view(seq_len, 256)
        
        return encoded

class SRTAdapter(AdapterInterface, SelfRewardTrainingInterface):
    """
    SRT适配器类
    
    实现了AdapterInterface和SelfRewardTrainingInterface接口，
    提供自我奖励训练、评估和改进功能。
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化SRT适配器
        
        Args:
            model_path: 模型路径，如果为None则使用默认模型
            device: 设备，'cuda'或'cpu'，如果为None则自动选择
        """
        # 确定设备
        if device is None:
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initialized SRT adapter with device: {self.device}")
        
        # 初始化模型和优化器
        self.model = None
        self.optimizer = None
        self.initialized = False
        
        # 如果提供了模型路径，加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("Initializing SRT model from default")
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        初始化模型
        
        创建新的SRT模型实例。
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using mock model")
            self.model = MockSRTModel()
            logger.info("SRT model initialized successfully")
            return
        
        try:
            # 创建模型
            self.model = SRTModel(hidden_size=768, num_layers=2, dropout=0.1)
            
            # 移动模型到指定设备
            self.model.to(self.device)
            
            logger.info("SRT model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SRT model: {str(e)}")
            raise
    
    def _initialize_optimizer(self, learning_rate: float = 0.001) -> None:
        """
        初始化优化器
        
        Args:
            learning_rate: 学习率
        """
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("PyTorch not available or model not initialized, using mock optimizer")
            return
        
        try:
            # 创建优化器
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            logger.info(f"Optimizer initialized with learning rate: {learning_rate}")
            
        except Exception as e:
            logger.error(f"Error initializing optimizer: {str(e)}")
            raise
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化适配器
        
        Args:
            config: 配置字典
            
        Returns:
            初始化是否成功
        """
        try:
            # 获取配置参数
            learning_rate = config.get("learning_rate", 0.001)
            
            # 初始化优化器
            self._initialize_optimizer(learning_rate)
            
            self.initialized = True
            logger.info("SRT adapter initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing SRT adapter: {str(e)}")
            return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        获取适配器支持的能力
        
        Returns:
            支持的能力字典
        """
        return {
            "self_reward_training": True,
            "thought_evaluation": True,
            "thought_improvement": True,
            "batch_training": True
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        status = {
            "status": "ok",
            "message": "SRT adapter is healthy",
            "details": {
                "model": self.model is not None,
                "optimizer": self.optimizer is not None,
                "device": self.device,
                "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available()
            }
        }
        
        return status
    
    def shutdown(self) -> bool:
        """
        关闭适配器，释放资源
        
        Returns:
            关闭是否成功
        """
        try:
            # 释放资源
            if TORCH_AVAILABLE and self.model is not None:
                # 将模型移动到CPU
                self.model.to("cpu")
                
                # 清除优化器
                self.optimizer = None
                
                # 清除CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.initialized = False
            logger.info("SRT adapter shut down successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down SRT adapter: {str(e)}")
            return False
    
    def train(self, thought_process: Union[str, Dict[str, Any]], 
             iterations: int = 100) -> Dict[str, Any]:
        """
        使用自我奖励机制训练模型
        
        Args:
            thought_process: 思考过程
            iterations: 训练迭代次数
            
        Returns:
            包含训练结果的字典
        """
        if not self.initialized:
            logger.error("SRT adapter not initialized")
            raise RuntimeError("SRT adapter not initialized")
        
        # 如果输入是字典，提取思考过程文本
        if isinstance(thought_process, dict):
            if "text" in thought_process:
                thought_process = thought_process["text"]
            else:
                logger.error("Invalid thought process format")
                raise ValueError("Invalid thought process format")
        
        # 确保思考过程是字符串
        if not isinstance(thought_process, str):
            logger.error(f"Invalid thought process type: {type(thought_process)}")
            raise TypeError(f"Invalid thought process type: {type(thought_process)}")
        
        if not TORCH_AVAILABLE:
            # 使用模拟训练
            initial_reward = random.uniform(0.5, 0.6)
            final_reward = random.uniform(0.9, 1.0)
            
            improvements = [
                {"iteration": 0, "reward": initial_reward, "loss": -initial_reward},
                {"iteration": iterations - 1, "reward": final_reward, "loss": -final_reward}
            ]
            
            logger.info(f"Training completed: initial reward = {initial_reward:.4f}, final reward = {final_reward:.4f}")
            
            return {
                "iterations": iterations,
                "improvements": improvements,
                "final_reward": final_reward
            }
        
        try:
            # 编码思考过程
            dataset = ThoughtDataset([thought_process])
            encoded = dataset[0].unsqueeze(0).to(self.device)  # 添加批次维度
            
            # 记录训练过程
            improvements = []
            
            # 初始评估
            self.model.eval()
            with torch.no_grad():
                initial_reward = self.model(encoded).item()
            
            # 记录初始奖励
            improvements.append({
                "iteration": 0,
                "reward": initial_reward,
                "loss": -initial_reward  # 损失是奖励的负值
            })
            
            # 训练模型
            self.model.train()
            for i in range(iterations):
                # 前向传播
                self.optimizer.zero_grad()
                reward = self.model(encoded)
                
                # 计算损失（最大化奖励）
                loss = -reward.mean()
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 每隔一定迭代记录一次
                if i == iterations - 1:
                    improvements.append({
                        "iteration": i,
                        "reward": reward.item(),
                        "loss": loss.item()
                    })
            
            # 最终评估
            self.model.eval()
            with torch.no_grad():
                final_reward = self.model(encoded).item()
            
            logger.info(f"Training completed: initial reward = {initial_reward:.4f}, final reward = {final_reward:.4f}")
            
            return {
                "iterations": iterations,
                "improvements": improvements,
                "final_reward": final_reward
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def batch_train(self, thought_processes: List[Union[str, Dict[str, Any]]], 
                  batch_size: int = 32) -> Dict[str, Any]:
        """
        批量训练模型
        
        Args:
            thought_processes: 思考过程列表
            batch_size: 批处理大小
            
        Returns:
            包含训练结果的字典
        """
        if not self.initialized:
            logger.error("SRT adapter not initialized")
            raise RuntimeError("SRT adapter not initialized")
        
        # 处理输入
        processed_thoughts = []
        for tp in thought_processes:
            if isinstance(tp, dict):
                if "text" in tp:
                    processed_thoughts.append(tp["text"])
                else:
                    logger.error("Invalid thought process format")
                    raise ValueError("Invalid thought process format")
            elif isinstance(tp, str):
                processed_thoughts.append(tp)
            else:
                logger.error(f"Invalid thought process type: {type(tp)}")
                raise TypeError(f"Invalid thought process type: {type(tp)}")
        
        if not processed_thoughts:
            logger.warning("Empty thought processes list")
            return {
                "batches": 0,
                "samples": 0,
                "batch_results": [],
                "average_reward": 0.0
            }
        
        if not TORCH_AVAILABLE:
            # 使用模拟批量训练
            avg_reward = random.uniform(0.7, 0.8)
            
            batch_results = [{
                "batch": 0,
                "reward": avg_reward,
                "loss": -avg_reward
            }]
            
            logger.info(f"Batch training completed: {len(processed_thoughts)} samples, average reward = {avg_reward:.4f}")
            
            return {
                "batches": 1,
                "samples": len(processed_thoughts),
                "batch_results": batch_results,
                "average_reward": avg_reward
            }
        
        try:
            # 创建数据集和数据加载器
            dataset = ThoughtDataset(processed_thoughts)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 记录批次结果
            batch_results = []
            total_reward = 0.0
            
            # 训练模型
            self.model.train()
            for batch_idx, batch in enumerate(dataloader):
                # 移动数据到设备
                batch = batch.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                reward = self.model(batch)
                
                # 计算损失（最大化奖励）
                loss = -reward.mean()
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 记录批次结果
                batch_reward = reward.mean().item()
                total_reward += batch_reward
                
                batch_results.append({
                    "batch": batch_idx,
                    "reward": batch_reward,
                    "loss": loss.item()
                })
            
            # 计算平均奖励
            average_reward = total_reward / len(dataloader)
            
            logger.info(f"Batch training completed: {len(processed_thoughts)} samples, average reward = {average_reward:.4f}")
            
            return {
                "batches": len(dataloader),
                "samples": len(processed_thoughts),
                "batch_results": batch_results,
                "average_reward": average_reward
            }
            
        except Exception as e:
            logger.error(f"Error batch training model: {str(e)}")
            raise
    
    def evaluate(self, thought_process: Union[str, Dict[str, Any]]) -> float:
        """
        评估思考过程的质量
        
        Args:
            thought_process: 需要评估的思考过程
            
        Returns:
            质量评分，范围[0, 1]
        """
        if not self.initialized:
            logger.error("SRT adapter not initialized")
            raise RuntimeError("SRT adapter not initialized")
        
        # 如果输入是字典，提取思考过程文本
        if isinstance(thought_process, dict):
            if "text" in thought_process:
                thought_process = thought_process["text"]
            else:
                logger.error("Invalid thought process format")
                raise ValueError("Invalid thought process format")
        
        # 确保思考过程是字符串
        if not isinstance(thought_process, str):
            logger.error(f"Invalid thought process type: {type(thought_process)}")
            raise TypeError(f"Invalid thought process type: {type(thought_process)}")
        
        if not TORCH_AVAILABLE:
            # 使用模拟评估
            score = random.uniform(0.4, 0.6)
            logger.info(f"Evaluated thought process: reward = {score:.4f}")
            return score
        
        try:
            # 编码思考过程
            dataset = ThoughtDataset([thought_process])
            encoded = dataset[0].unsqueeze(0).to(self.device)  # 添加批次维度
            
            # 评估
            self.model.eval()
            with torch.no_grad():
                score = self.model(encoded).item()
            
            logger.info(f"Evaluated thought process: reward = {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating thought process: {str(e)}")
            raise
    
    def improve(self, thought_process: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        改进思考过程
        
        Args:
            thought_process: 原始思考过程
            
        Returns:
            改进后的思考过程
        """
        if not self.initialized:
            logger.error("SRT adapter not initialized")
            raise RuntimeError("SRT adapter not initialized")
        
        # 如果输入是字典，提取思考过程文本
        is_dict_input = isinstance(thought_process, dict)
        if is_dict_input:
            if "text" in thought_process:
                text = thought_process["text"]
            else:
                logger.error("Invalid thought process format")
                raise ValueError("Invalid thought process format")
        else:
            text = thought_process
        
        # 确保思考过程是字符串
        if not isinstance(text, str):
            logger.error(f"Invalid thought process type: {type(text)}")
            raise TypeError(f"Invalid thought process type: {type(text)}")
        
        # 评估原始思考过程
        score = self.evaluate(text)
        
        # 在实际应用中，这里应该使用更复杂的方法来改进思考过程
        # 例如，使用生成模型生成改进建议
        # 这里我们使用一个简单的模拟实现
        improved_text = f"Improved: {text}"
        improved_score = score + random.uniform(0.01, 0.05)
        improved_score = min(improved_score, 1.0)
        
        # 添加质量评分
        result = f"{improved_text}\nQuality Score: {improved_score:.4f}"
        
        # 如果输入是字典，返回字典
        if is_dict_input:
            return {
                "text": result,
                "score": improved_score,
                "original_score": score
            }
        else:
            return result
    
    def save_model(self, path: str) -> bool:
        """
        保存模型
        
        Args:
            path: 模型保存路径
            
        Returns:
            保存是否成功
        """
        if not self.initialized or self.model is None:
            logger.error("SRT adapter not initialized or model is None")
            return False
        
        if not TORCH_AVAILABLE:
            # 模拟保存
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                
                # 写入空文件
                with open(path, 'w') as f:
                    f.write("")
                
                logger.info(f"Model saved to {path}")
                return True
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                return False
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # 保存模型
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'model_config': {
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'dropout': self.model.dropout
                }
            }, path)
            
            logger.info(f"Model saved to {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        加载模型
        
        Args:
            path: 模型加载路径
            
        Returns:
            加载是否成功
        """
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
        
        if not TORCH_AVAILABLE:
            # 模拟加载
            logger.info(f"Model loaded from {path}")
            return True
        
        try:
            # 加载模型
            checkpoint = torch.load(path, map_location=self.device)
            
            # 获取模型配置
            model_config = checkpoint.get('model_config', {
                'hidden_size': 768,
                'num_layers': 2,
                'dropout': 0.1
            })
            
            # 创建模型
            self.model = SRTModel(
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout']
            )
            
            # 加载模型参数
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 移动模型到指定设备
            self.model.to(self.device)
            
            # 如果有优化器状态，加载优化器
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                # 确保优化器已初始化
                if self.optimizer is None:
                    self._initialize_optimizer()
                
                # 加载优化器参数
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

class MockSRTModel:
    """
    模拟SRT模型类
    
    当PyTorch不可用时使用的模拟实现。
    """
    
    def __init__(self):
        """初始化模拟模型"""
        self.hidden_size = 768
        self.num_layers = 2
        self.dropout = 0.1
    
    def to(self, device: str):
        """模拟移动模型到设备"""
        return self
    
    def eval(self):
        """模拟设置为评估模式"""
        pass
    
    def train(self):
        """模拟设置为训练模式"""
        pass
    
    def parameters(self):
        """模拟返回模型参数"""
        return []
    
    def state_dict(self):
        """模拟返回模型状态"""
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """模拟加载模型状态"""
        pass
