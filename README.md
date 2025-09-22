# AI代理驱动的代码空间探索框架

## 🎯 项目概述

本项目是Hung-yi Lee ML2025-hw2的实现，并做了一些修改，使其支持调用api来完成任务。受到AIDE论文 ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)) 的启发，实现了一个**Agent代码生成与迭代优化**系统。该框架将编程任务建模为**代码起草和迭代优化问题**，通过**树搜索算法**在潜在解决方案空间中进行智能探索，得到最优的代码。



## 🏗️ 核心架构

### 智能代理系统设计
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Agent Core    │───▶│  Journal State   │───▶│  Node Tree      │
│  (决策引擎)      │    │   (状态管理)      │    │  (解决方案树)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  LLM Client     │    │  Interpreter     │    │  Evaluation     │
│  (代码生成)      │    │  (代码执行)       │    │  (性能评估)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 三阶段解决方案演化

```
任务输入 → Draft阶段 → 代码执行与评估
            ↓
         生成初始方案
            ↓
    ┌─── 执行成功？ ───┐
    ↓                ↓
   YES              NO
    ↓                ↓
Improve阶段      Debug阶段
    ↓                ↓
基于最佳方案优化   分析错误并修复
    ↓                ↓
 性能提升版本    ← 修复版本
    ↓                ↓
   代码执行与评估 ←───────┘
    ↓
┌─ 达到最大步数？ ─┐
↓                ↓
YES              NO
↓                ↓
输出最佳方案    ←─┘
```

#### 阶段详解：
- **Draft**: 从零开始创建解决方案，重点是完整性和可执行性
- **Debug**: 错误驱动的修复过程，重点是正确性和稳定性  
- **Improve**: 性能驱动的优化过程，重点是效果和效率


### 技术组件详解

| 组件 | 功能 | 核心特性 |
|------|------|----------|
| **Agent** | 智能决策引擎 | 搜索策略、方案生成、迭代优化 |
| **Journal** | 状态管理系统 | 解决方案树、性能跟踪、历史记录 |
| **Node** | 解决方案节点 | 代码+计划+结果、父子关系、阶段标识 |
| **Interpreter** | 代码执行引擎 | 多进程隔离、代码执行、异常捕获 |
| **LLMClient** | 智能代码生成 | 自然语言理解、代码生成、错误修复 |

## 🎯 示例应用场景

该项目是本框架用于机器学习任务的示例，在完成任务描述后使用本系统生成了一个监督学习中的回归任务的代码。使用的数据集包含美国34个州、连续3天COVID-19调查数据（症状、行为、信念等77个特征），预测第3天的测试阳性概率。AI代理需要自动进行特征工程、模型选择和超参数优化，处理时间序列和地理分布的复杂性。（框架可适用于任何需要代码生成和迭代优化的编程任务）

## 🧠 智能决策机制

### 搜索策略算法
```python
def search_policy(self) -> Node | None:
    # 1. 确保有足够的初始草稿
    if len(self.journal.draft_nodes) < search_cfg.num_drafts:
        return None
    
    # 2. 概率性选择调试路径
    if random.random() < search_cfg.debug_prob:
        debuggable_nodes = [n for n in self.journal.buggy_nodes if n.is_leaf]
        if debuggable_nodes:
            return random.choice(debuggable_nodes)
    
    # 3. 选择最佳节点进行改进
    good_nodes = self.journal.good_nodes
    if not good_nodes:
        return None
    return self.journal.get_best_node()
```


## 🚀 快速开始

### 环境配置
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置API密钥
cp .env.example .env
# 编辑 .env 文件，设置你的 DEEPSEEK_API_KEY
```

### 运行系统
```bash
python main.py
```

### 自定义任务
修改 `config.py` 中的配置：
```python
config = {
    "task_goal": "你的ML任务描述",
    "data_dir": Path("./your_dataset").resolve(),
    "agent": {
        "steps": 5,  # 代理执行步数
        "search": {
            "debug_prob": 0.5,  # 调试概率
            "num_drafts": 1,    # 初始草稿数量
        },
    },
}
```

## 💡 核心创新点

### 🎲 **自适应搜索策略**
- 基于概率的智能决策
- 动态平衡探索与利用
- 自动选择最优改进路径

### 🌳 **解决方案演化树**
- 维护完整的方案演化历史
- 支持回溯和分支探索
- 智能选择最佳节点进行扩展

### 🛡️ **安全代码执行**
- 多进程隔离执行环境
- 完整的异常捕获和处理
- 超时控制和资源管理

### 🔄 **端到端自动化**
- 从任务理解到代码生成
- 自动执行和结果评估
- 基于反馈的迭代优化

## 📁 项目结构

```
├── agent.py              # 🎯 核心AI代理逻辑
│   ├── search_policy()   #    智能搜索策略
│   ├── _draft()         #    生成初始方案
│   ├── _debug()         #    错误修复
│   └── _improve()       #    方案优化
├── models.py             # 📊 数据模型定义
│   ├── Node             #    解决方案节点
│   └── Journal          #    状态管理器
├── interpreter_module.py # ⚡ 安全代码执行引擎
├── llm_client.py         # 🤖 LLM接口封装
├── text_utils.py         # 📝 文本处理工具
├── utils.py              # 🛠️ 通用工具函数
├── config.py             # ⚙️ 系统配置
├── main.py               # 🚀 程序入口
└── dataset/              # 📊 数据集目录
```

## 🔧 高级配置

### 代理行为调优
```python
# 在 config.py 中调整
"agent": {
    "steps": 10,           # 增加执行步数获得更好结果
    "search": {
        "debug_prob": 0.3, # 降低调试概率，更多改进
        "num_drafts": 2,   # 增加初始方案多样性
    },
}
```

### 自定义评估指标
系统当前使用MSE作为评估指标，可在代码中扩展其他指标。

## 🎯 应用场景

### 机器学习工程自动化
- ✅ **Kaggle竞赛**: 自动化特征工程、模型选择和超参数优化
- ✅ **ML原型开发**: 快速验证算法想法和实验假设
- ✅ **基准测试**: 在MLE-Bench、METR等评估基准上的自动化实验

### 代码空间探索研究
- ✅ **算法研究**: 研究不同搜索策略在代码优化中的效果
- ✅ **教育工具**: 理解AI代理如何进行系统性的试错和优化
- ✅ **框架扩展**: 作为AIDE类系统的研究和开发基础

### 通用编程任务
- ✅ **自动化脚本生成**: 数据处理、文件操作等重复性编程任务
- ✅ **代码优化**: 性能优化、重构建议等代码改进任务



## 🚀 扩展开发

### 添加新的搜索策略
```python
def custom_search_policy(self) -> Node | None:
    # 实现你的搜索逻辑
    pass
```

### 集成新的LLM
```python
class CustomLLMClient:
    def generate_response(self, messages):
        # 实现你的LLM接口
        pass
```

### 支持新的评估指标
```python
def custom_evaluation(predictions, targets):
    # 实现你的评估逻辑
    return metric_value
```

## 📈 性能特点

- **智能化程度高**: 自主决策，无需人工干预
- **扩展性强**: 模块化设计，易于定制和扩展
- **安全可靠**: 隔离执行，完整的错误处理
- **通用性好**: 适用于各种ML任务场景

## 🤝 贡献指南

欢迎提交Issue和Pull Request！特别欢迎：
- 新的搜索策略实现
- 更多的评估指标支持
- 性能优化建议
- 文档改进

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 📖 引用

如果这个项目对你的研究有帮助，请考虑引用原始的AIDE论文：

```bibtex
@article{jiang2025aide,
  title={AIDE: AI-Driven Exploration in the Space of Code},
  author={Jiang, Zhengyao and Schmidt, Dominik and Srikanth, Dhruv and Xu, Dixing and Kaplan, Ian and Jacenko, Deniss and Wu, Yuxiang},
  journal={arXiv preprint arXiv:2502.13138},
  year={2025}
}
```

---
