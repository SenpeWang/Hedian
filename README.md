# 核电站监护制合规检测系统 (Hedian)

核电站主控室场景下的**多模态合规性实时检测系统**。系统分析正面（front）与操作盘（pop）两路音视频，自动检测操作人/监护人是否按规程执行监护制、自唱票、信息通报等制度，并通过大模型对流程合规性进行自动评估，前端流式可视化展示。

## ✨ 功能特性

- **双路视频实时推理**：front（30fps，带音频）+ pop（25fps）两路视频并行分析
- **多模态感知**：
  - 人员追踪（Tracker）：目标检测 + 姿态估计 + 多目标跟踪
  - 眼睛关注度（Gaze）：头部姿态 + 视线估计，判断是否注视关键区域
  - 语音转录（Voice）：Qwen3-ASR 字词级对齐 + 行业术语归一化
  - 行为检测（Behavior）：举手、手指屏幕、手指文件
- **规程合规判定**：监护制/自唱票/信息通报/人员状态 四大状态机实时研判 + 违规告警
- **大模型合规评估**：Qwen3-8B 对完整流程多模态证据链自动评估打分
- **前后端流式可视化**：视频、字幕、关注度热力图、告警实时推送

## 📁 目录结构

```
├── main.py                 # 入口：启动推理子进程 + Web 常驻（端口 5002）
├── config.yaml             # 视角级 GPU / Redis / 路径 / 视频配置
├── core/                   # 核心框架（事件总线/推理流/同步对齐/可视化编码）
├── modules/                # 推理层（voice / tracker(+gaze) / behavior）
├── evaluation/             # 评估层（大模型评估/数据提取/流程管理）
├── rules/                  # 规程状态机（监护制/自唱票/信息通报/人员状态）
├── web/                    # Web 层（HTTP / WebSocket / 可视化转发）
├── frontend/               # Vue3 前端（流式播放/状态管理/组件）
├── models/                 # 模型权重（gitignore，不入库）
└── data/                   # 视频与结果（gitignore，不入库）
```

## 🚀 快速开始

### 环境要求

- Python **3.10**（conda 环境 `sp_hedian`）
- Redis（Stream + Hash）
- NVIDIA GPU（RTX 4090 ×2，视角级 GPU 分配）
- Node **20**（前端构建用）

### 1. 进入项目目录

```bash
cd /hedian
```

### 2. 配置虚拟环境

```bash
conda create -n sp_hedian python=3.10
conda activate sp_hedian
pip install -r requirements.txt
```

### 3. 启动后端

```bash
cd /hedian
setsid nohup python main.py --gpu 0 > /tmp/hedian_web.log 2>&1 < /dev/null &
```

### 4. 构建前端（需 Node 20）

```bash
cd /hedian/frontend
npm run build
```

### 5. 访问

浏览器打开 `http://<服务器IP>:5002`

### 停止服务

```bash
curl -X POST http://127.0.0.1:5002/stop
pgrep -f main.py | xargs kill -9
```

## 🧠 技术栈

| 维度 | 技术 |
|------|------|
| 后端 | Python 3.10 / FastAPI + uvicorn / Redis Stream+Hash / ffmpeg fMP4 |
| 前端 | Vue3 + TypeScript + Vite / MSE (MediaSource Extensions) 流式 |
| 追踪 | yolo11l 检测 + yolo26s-pose 姿态 + OC-SORT 多目标跟踪 |
| 语音 | Qwen3-ASR-0.6B + ForcedAligner 对齐 |
| 评估 | Qwen3-8B 大模型合规评估 |
| 硬件 | RTX 4090 ×2（tracker→GPU0，voice/behavior/evaluation→GPU1） |

## 🔗 文档

- `docs/`：SPEC（需求）、ARCHITECTURE（设计）、API（接口契约）、REQUIREMENTS（验收标准）
- `docs/rule_doc/`：监护制/自唱票/信息通报 判定逻辑权威参考

## 📄 License

Internal project.
