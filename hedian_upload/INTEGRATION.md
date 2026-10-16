# 信息通报规程 · 代码整合说明

> 本文件用于指导把「信息通报规程」相关代码整合进 **Hedian** 仓库，供上传者与接收者参考。
> 文档本身不含业务逻辑，上传仓库时可自行剔除。

---

## 一、这批文件是什么

来自 `A_DemoSrc` 工作区中**已开发完成、尚未提交**的信息通报改动，覆盖从感知到前端展示的 6 层，实现了完整的信息通报合规检测能力：

- **规则状态机**：信息通报五步判定 —— 举手并喊话发起 / 停下手中工作 / 团队关注 / 喊出通报完毕 / 回答收到
- **实时子步骤下发**：每项判定在观测当场经 `RULE_FLOW_STEP` 推送到前端「系统通知」面板，逐行显示（判定与合规与否），不做收口后批处理
- **10 秒团队关注度窗口**：规则层发布 `FLOW_STARTED(flow_type=info_notice)` 后，gaze 模块按 `flow_id` 开启 10s 注视采样，结束下发 `GAZE_ATTENTION`
- **证据到齐再收口**：ASR 远快于画面推理，收口前先挂起等待关注度结论（`EVIDENCE_WAIT_SEC`），避免把"尚未算出的关注度"误判为未关注
- **评估层**：Qwen3-8B 负责成文；团队总分（5 项 × 2 分）以规则层的确定性判定为准
- **前端**：系统通知面板子步骤行、流程评价卡片；引入 `flowKey` 修复 `info_notice#1` 与 `supervision#1` 的 `flow_id` 撞车

---

## 二、文件清单（共 32 个）

### 1. 规则层 `rules/`
| 文件 | 作用 |
|---|---|
| `info_notice_rule.py` | **信息通报状态机（核心，完全重写）** |
| `rule_base.py` | 基类 `finalize` 增加 `final_timestamp` 参数 |
| `flow_recorder.py` | 流程事件记录，修复 flow_id 撞车 |
| `__init__.py` | 注册 `InfoNoticeRule` |
| `supervision_rule.py` | 适配 speech_transcriber 新接口 + `finalize` 签名 |
| `personnel_status_rule.py` | `finalize` 签名对齐 |

### 2. 核心层 `core/`
| 文件 | 作用 |
|---|---|
| `event_bus.py` | 新增 `RULE_FLOW_STEP` 事件主题 |
| `inference_sync.py` | 订阅 `RULE_FLOW_STEP`，推送 `flow_step` 给前端 |
| `config_manager.py` | voice 默认参数（`sentence_gap_sec` / `keyword_span_sec`） |

### 3. 感知模块 `modules/`
| 文件 | 作用 |
|---|---|
| `gaze/gaze_module.py` | **信息通报 10 秒关注度窗口 + 逐工位判定** |
| `tracker/tracker_module.py` | 注入 run_id 给 gaze（关注度中途落盘） |
| `voice/voice_module.py` | 发布 `VOICE_KEY_MOMENT` 事件 |
| `voice/speech_transcriber.py` | 语音转录 + 关键词识别（含信息通报关键词） |
| `voice/sentence_assembler.py` | 跨段词流组装，保护词防断（"通报完毕""收到"等） |
| `voice/segmenters.py` | VAD 在线切片（流式管线） |
| `voice/streaming_source.py` | 流式/直播源读取 |
| `voice/system_code_corrector.py` | 九字码结构矫正（被 speech_transcriber 顶层 import） |

### 4. 配置与评估
| 文件 | 作用 |
|---|---|
| `configs/system_code_rules.json` | 九字码码表（system_code_corrector 依赖） |
| `evaluation/qwen_evaluator.py` | 信息通报五维度评估 prompt |
| `evaluation/flow_data_extractor.py` | 按 `evidence_end_sec` 提取证据 |
| `evaluation/flow_evaluation_manager.py` | info_notice 评估调度 + 总分以规则层为准 |

### 5. 入口与文档
| 文件 | 作用 |
|---|---|
| `main.py` | 配置回灌 + `finalize` 传视频时长 |
| `config.yaml` | `rules.info_notice` / voice 段配置 |
| `README.md` | 文档更新（见第五节注意事项 3） |

### 6. 前端 `frontend/`
| 文件 | 作用 |
|---|---|
| `src/types.ts` | `FlowType` / `FlowKind` / `flowKey` 类型 |
| `src/App.vue` | `flow_step` 推送处理 |
| `src/style.css` | `flow-event--step` 子步骤行样式 |
| `src/components/NotifyPanel.vue` | 系统通知面板 |
| `src/components/ReportPanel.vue` | 流程评价面板 |
| `src/composables/useNotify.ts` | 通知行组装（流程序号、去重） |
| `src/composables/useReports.ts` | 评价卡片（info_notice 修复） |
| `src/composables/useTypewriter.ts` | 打字机改用 `flowKey` |

---

## 三、目录结构

```
.
├── config.yaml
├── main.py
├── README.md
├── configs/system_code_rules.json
├── core/{config_manager.py, event_bus.py, inference_sync.py}
├── evaluation/{qwen_evaluator.py, flow_data_extractor.py, flow_evaluation_manager.py}
├── modules/
│   ├── gaze/gaze_module.py
│   ├── tracker/tracker_module.py
│   └── voice/{voice_module.py, speech_transcriber.py, sentence_assembler.py,
│              segmenters.py, streaming_source.py, system_code_corrector.py}
├── rules/{info_notice_rule.py, rule_base.py, flow_recorder.py, __init__.py,
│          supervision_rule.py, personnel_status_rule.py}
└── frontend/src/
    ├── {App.vue, types.ts, style.css}
    ├── components/{NotifyPanel.vue, ReportPanel.vue}
    └── composables/{useNotify.ts, useReports.ts, useTypewriter.ts}
```

> 所有路径均相对**仓库根目录**，与 Hedian 现有目录一一对应。

---

## 四、整合步骤

### 方式 A：GitHub 网页上传

1. 打开 Hedian 仓库 → **Add file → Upload files**
2. 把本文件夹内的所有内容拖入（GitHub 会保留内部目录结构），填写提交信息后提交
3. **手动删除** `rules/self_ticket_rule.py`：打开该文件 → 右上角垃圾桶 → 提交
   （网页上传只能新增/覆盖，无法删除远程文件）

### 方式 B：git 命令行（推荐）

```bash
# 1. 克隆仓库（若已有本地仓库，先拉取最新）
git clone git@github.com:SenpeWang/Hedian.git
cd Hedian
git pull --rebase            # 拉取远程最新，避免落后

# 2. 用本文件夹的内容覆盖到仓库（保持同名路径）
cp -r /path/to/hedian_upload/* .

# 3. 暂存全部改动；删除自唱票规则
git add -A
git rm rules/self_ticket_rule.py

# 4. 提交并推送
git commit -m "feat(rules): 完善信息通报规程 - 五步实时判定/10s关注度窗口/子步骤下发"
git push origin main
```

---

## 五、注意事项（重要）

### 1. 这些文件必须**整体**提交，不可拆分
文件之间存在硬依赖，缺任何一个都会导致 `import` 失败或功能缺失：

- `voice_module.py` / `speech_transcriber.py` → 依赖 `segmenters.py`、`streaming_source.py`、`sentence_assembler.py`、`system_code_corrector.py`
- `system_code_corrector.py` → 依赖 `configs/system_code_rules.json`
- `rule_base.py` 的 `finalize` 签名变化 → 连带 `supervision_rule.py`、`personnel_status_rule.py` 必须同步
- `speech_transcriber.py` 导出接口变化 → 连带 `supervision_rule.py` 的 import 适配
- 前端 `flowKey` 字段 → 连带 `useTypewriter.ts` 等

### 2. `rules/self_ticket_rule.py` 需要删除
这属于"移除自唱票功能"，与信息通报是两件独立的事。若希望分开管理，可拆成单独一次提交；但**务必删除**，否则 `rules/__init__.py` 已不再导出它，且 `config.yaml` 也已移除 `self_ticket` 开关。

### 3. `README.md` 中含本地环境信息，上传前请精简
原 README 的改动里混入了本地演示相关内容（`config.5009.yaml`、`新建文件夹/信息通报_*.mp4`、`/home/...` 等绝对路径），不适合出现在公开仓库，建议在上传前删掉这些行。

### 4. 前端需要重新构建
本文件夹只含**前端源码**（`frontend/src`），不含构建产物。接收者需自行构建：

```bash
cd frontend
npm install
npm run build
```

### 5. 以下文件/目录**不要**提交
```
config.5008.yaml / config.5009.yaml / restart_5009.sh   # 本地环境配置
frontend/dist/**  frontend/tsconfig.tsbuildinfo          # 构建产物
data/  models/                                            # 本地资源（应在 .gitignore）
core/redis_conn.py  web/http_server.py                    # Redis 清理重构，与本功能无关
modules/behavior/behavior_utils.py                        # 工具方法，无关
frontend/src/composables/{usePlayback,useTranscript}.ts   # 通用前端改进，无关
requirements.txt 中的 websockets                          # web 服务依赖，无关
```

---

## 六、验证清单

整合完成后，建议按顺序验证：

1. **后端启动**：`python main.py --gpu 0`，确认无 `ModuleNotFoundError`
2. **模块注册**：日志出现 `发现制度: info_notice`
3. **规则链路**：跑一段含信息通报的视频，观察日志
   - `信息通报流程开始 flow_id=...`
   - `信息通报: 步骤下发[...]`
   - `GazeModule: 信息通报 10S 关注度评估完成`
   - `信息通报流程结束 ... 得分=X/10`
4. **前端面板**：播放到流程时刻，「系统通知」面板逐行出现子步骤，「流程评价」出现信息通报卡片（分数与规则层一致）
5. **事件总线**：确认 Redis 中 `module:events:rule.flow_step` 有数据

---

*本说明由 A_DemoSrc 工作区整理生成，如与代码实际行为不符，以代码为准。*
