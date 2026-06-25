# 核电站行为合规检测系统

## 项目概述

核电站监护制合规检测系统，通过视频和音频分析，自动检测操作人员是否遵守监护制和自唱票规程。

## 环境

- **服务器**: 10.152.57.223 (SSH: wangshengping / Wsp991225)
- **Python**: /home/wangshengping/myconda/envs/sp_hedian/bin/python
- **GPU**:  RTX 4090 (24GB)
- **启动命令**: `python -u main.py --gpu 0`

## 目录结构

```
A_DemoSrc/
├── main.py                    # 入口文件
├── config.yaml                # 全局配置
├── CLAUDE.md                  # 本文件
├── core/                      # 核心框架
│   ├── message_bus.py         # 消息总线（模块间通信）
│   ├── config_manager.py      # 配置管理
│   ├── event_aggregator.py    # 事件时间对齐 + SSE 推送
│   ├── base_module.py         # 模块基类
│   ├── path_manager.py        # 路径管理
│   └── logger.py              # 日志系统
├── modules/                   # 业务模块
│   ├── mot/                   # 多目标跟踪（检测 + 跟踪 + 举手 + 监护状态）
│   ├── gaze/                  # 注视检测（头部检测 + 注视推断 + ROI 分类）
│   ├── voice/                 # 语音转文字（Whisper ASR + 意图分类）
│   └── behavior/              # 行为检测（姿态 + 手指指向屏幕）
├── regulations/               # 制度层（监护制、自唱票规则）
├── evaluation/                # 评估层（规则评估、大模型评估）
├── web/                       # 前端层
│   ├── http_server.py         # Flask 路由（/ /start /events /stream /stream2 /audio）
│   ├── sse_handler.py         # SSE 推送
│   └── static/index.html      # 前端页面
├── models/                    # 模型文件
│   ├── detection/             # yolo11_MOT.pt, yolo26s-pose.pt
│   ├── gaze/                  # yolov8n_head.onnx, gazelle_*.onnx
│   ├── behavior/              # yolo11l-pose.pt, yolov8_finger.pt
│   └── voice/                 # large-v3.pt (Whisper)
└── data/                      # 数据目录
    ├── videos/                # 视频文件
    ├── ROI.json               # 注视区域配置（LabelMe 格式）
    └── results/               # 运行结果
```

## 架构设计

### 消息总线模式
- 所有模块通过 `MessageBus` 发布/订阅事件
- 消息类型定义在 `MsgType` 常量中
- 模块间松耦合，可独立启停

### 事件聚合器
- `EventAggregator` 负责多模块时间对齐
- 支持即时事件（status, progress, video_start, done）
- 通过 SSE 推送到前端

### 模块基类
所有业务模块继承 `BaseModule`，实现统一接口：
- `module_name`: 模块名称
- `initialize()`: 初始化模型
- `process_video(video_path)`: 处理视频
- `save_results(run_id)`: 保存结果

## 关键配置 (config.yaml)

```yaml
modules:
  voice: true          # 语音模块开关
  mot: true            # 目标跟踪模块开关
  gaze: true           # 注视检测模块开关
  behavior: true       # 行为检测模块开关

paths:
  video: data/videos/camFRONT.mpg

supervision:
  dist_close_px: 280   # 监护距离阈值（像素）
```

## 前端

- 端口: 5000
- 路由: `/` (首页), `/start` (GET 启动), `/events` (SSE), `/stream` (MOT视频流), `/stream2` (行为视频流)
- 实时显示: 语音转写、目标跟踪、行为检测、评价报告

## 运行

```bash
# SSH 连接
ssh wangshengping@10.152.57.223

# 启动服务
cd /home/wangshengping/Hedian/A_DemoSrc
/home/wangshengping/myconda/envs/sp_hedian/bin/python -u main.py --gpu 0

# 或使用 tmux
tmux new-session -d -s hedian '/home/wangshengping/myconda/envs/sp_hedian/bin/python -u main.py --gpu 0 2>&1 | tee /tmp/hedian.log'
```

