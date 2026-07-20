"""
大模型评估模块
"""
import os
import json
import logging
import multiprocessing as mp
from typing import Dict

logger = logging.getLogger("evaluation.qwen")

# 设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _qwen_worker(model_path: str, prompt: str, queue):
    """子进程工作函数：在物理 GPU 0 上加载模型并生成评估报告

    子进程设置 CUDA_VISIBLE_DEVICES=0，使 cuda:0 映射到物理 GPU 0。
    评估完成后子进程退出，显存自动释放。
    """
    # 必须在 import torch 前设置 CPU 核心使用限制与 GPU 绑定，避免抢占 Web 线程 CPU 造成声音卡顿
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        import torch
        torch.set_num_threads(1)
        import threading
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

        logging.getLogger("evaluation.qwen").info(
            f"子进程加载 Qwen3 模型: {model_path} (物理 GPU 0, bfloat16)"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            local_files_only=True,
        )

        try:
            allocated = torch.cuda.memory_allocated("cuda:0") / 1024**3
            logging.getLogger("evaluation.qwen").info(
                f"Qwen3 模型加载完成，物理 GPU 0 显存占用: {allocated:.2f} GB"
            )
        except Exception:
            pass

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 8192,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            queue.put(("chunk", text))

        thread.join()
        queue.put(("done", None))

    except Exception as e:
        import traceback
        queue.put(("error", f"{e}\n{traceback.format_exc()}"))


class QwenEvaluator:
    """Qwen 大模型评估器（子进程模式，物理 GPU 0，评估完自动释放显存）"""

    def __init__(self, model_path: str = None):
        self._model_path = model_path

    def evaluate(self, flow_data: Dict, stream_callback=None, total_flows: int = 0) -> Dict:
        """在独立子进程中评估流程（物理 GPU 0），评估完子进程退出释放显存"""
        p = None
        prompt = ""
        try:
            prompt = self._build_prompt(flow_data, total_flows=total_flows)

            logger.info(
                f"开始 Qwen 评估（子进程，物理 GPU 0），"
                f"flow_id={flow_data.get('flow_id')} total_flows={total_flows}..."
            )

            queue = mp.Queue()
            p = mp.Process(
                target=_qwen_worker,
                args=(self._model_path, prompt, queue),
            )
            p.start()

            report_text = ""
            while p.is_alive() or not queue.empty():
                try:
                    msg_type, data = queue.get(timeout=0.2)
                    if msg_type == "chunk":
                        report_text += data
                        if stream_callback:
                            stream_callback(data)
                    elif msg_type == "done":
                        break
                    elif msg_type == "error":
                        logger.error(f"子进程评估失败: {data}")
                        return {
                            "flow_type": flow_data.get("flow_type", "未知"),
                            "score": 0,
                            "report_text": f"评估失败: {data}",
                            "prompt": prompt,
                        }
                except Exception:
                    pass

            p.join(timeout=5)
            if p.is_alive():
                logger.warning("子进程未正常退出，强制终止")
                p.terminate()
                p.join(timeout=3)

            score = self._extract_score(report_text)
            logger.info(f"Qwen 评估完成，评分: {score}")

            return {
                "flow_type": flow_data.get("flow_type", "未知"),
                "score": score,
                "report_text": report_text,
                "prompt": prompt,
            }

        except Exception as e:
            logger.error(f"Qwen 评估失败: {e}", exc_info=True)
            return {
                "flow_type": flow_data.get("flow_type", "未知"),
                "score": 0,
                "report_text": f"评估失败: {e}",
                "prompt": prompt,
            }
        finally:
            if p is not None:
                try:
                    if p.is_alive():
                        logger.warning("清理：评估子进程仍在运行，强制终止")
                        p.terminate()
                        p.join(timeout=3)
                    else:
                        p.join(timeout=1)
                except Exception as ex:
                    logger.error(f"清理评估子进程出错: {ex}")
                try:
                    p.close()
                except Exception:
                    pass

    def _build_prompt(self, flow_data: Dict, total_flows: int = 0) -> str:
        """构建评估 prompt"""
        flow_type = flow_data.get("flow_type", "未知")
        flow_id = flow_data.get("flow_id", 0)

        flow_type_cn_map = {
            "supervision": "监护制",
            "self_ticket": "自唱票",
            "info_notice": "信息通报",
        }
        flow_type_cn = flow_type_cn_map.get(flow_type, flow_type)

        if total_flows > 0:
            flow_data = dict(flow_data)
            flow_data["_total_flows"] = total_flows
            flow_data["_flow_id_desc"] = (
                f"本次共 {total_flows} 个{flow_type_cn}流程，"
                f"当前为第 {flow_id} 个{flow_type_cn}流程"
            )

        if flow_type == "supervision":
            return self._build_supervision_prompt(flow_data)
        elif flow_type == "self_ticket":
            return self._build_self_ticket_prompt(flow_data)
        elif flow_type == "info_notice":
            return self._build_info_notice_prompt(flow_data)
        else:
            return f"评估流程类型: {flow_type}"

    def _build_info_notice_prompt(self, flow_data: Dict) -> str:
        """构建信息通报评估 prompt"""
        voice_events = flow_data.get("voice_events", [])
        gaze_events = flow_data.get("gaze_events", [])
        behavior_events = flow_data.get("behavior_events", [])

        parts = []
        parts.append("你是核电站主控室信息通报规程合规检测评审专家。请根据以下流程数据，按五个评估维度进行评分。")
        parts.append("")
        parts.append("## 信息通报规程规则")
        parts.append("1. 流程启动：发起人举手 + 喊出\"信息通报\"或\"信息通告\"（红线规则：单独举手不启动流程）")
        parts.append("2. 团队接受：主控室其他成员听到后立刻停下手中工作")
        parts.append("3. 团队关注：发起人确认团队成员均予以关注，方可进行信息传递")
        parts.append("4. 信息结束：发起人喊出\"通报完毕\"或\"通告完毕\"（即时闭环，不等待\"收到\"回复）")
        parts.append("5. 值长回应：值长（US）回答\"收到\"（作为评估依据，不阻塞流程关闭）")
        parts.append("")
        parts.append("## 流程信息")
        parts.append(f"- 流程类型: 信息通报")
        if flow_data.get('_flow_id_desc'):
            parts.append(f"- {flow_data.get('_flow_id_desc')}")
        parts.append(f"- 开始时间: {flow_data.get('flow_start_sec', 0)}s")
        parts.append(f"- 结束时间: {flow_data.get('flow_end_sec', 0)}s")
        parts.append(f"- 持续时间: {flow_data.get('flow_continue_sec', 0)}s")
        parts.append(f"- 触发来源: {flow_data.get('start_source', '未知')}")
        parts.append(f"- 结束来源: {flow_data.get('end_source', '未知')}")
        parts.append("")
        parts.append("## 语音事件（按时间排序）")
        for event in voice_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        if not voice_events:
            parts.append("- 无")
        parts.append("")
        parts.append("## 注视事件（团队关注度判定）")
        for event in gaze_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        if not gaze_events:
            parts.append("- 无")
        parts.append("")
        parts.append("## 行为事件（举手等）")
        for event in behavior_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        if not behavior_events:
            parts.append("- 无")
        parts.append("")
        parts.append("## 评估维度（每项2分，满分10分）")
        parts.append("| 维度 | 评估要点 | 评分标准 |")
        parts.append("|------|---------|---------|")
        parts.append("| 1. 流程启动 | 发起人是否举手 + 喊出\"信息通报\" | 2分=举手+语音；1分=仅语音；0分=无启动信号 |")
        parts.append("| 2. 团队关注 | 团队成员是否予以关注 | 2分=有视线转动；1分=部分关注；0分=无关注 |")
        parts.append("| 3. 信息结束 | 是否喊出\"通报完毕\" | 2分=有\"通报完毕\"；0分=无 |")
        parts.append("| 4. 值长回应 | 值长是否回答\"收到\" | 2分=有\"收到\"；0分=无 |")
        parts.append("| 5. 整体规范 | 流程完整度与时间合理性 | 2分=完整规范；1分=基本规范；0分=不规范 |")
        parts.append("")
        parts.append("## 输出格式")
        parts.append("请按以下格式输出评估报告：")
        parts.append("")
        parts.append("### 信息通报评估报告")
        parts.append("")
        parts.append("#### 维度评分")
        parts.append("- 流程启动: X分/2分 - [依据与说明]")
        parts.append("- 团队关注: X分/2分 - [依据与说明]")
        parts.append("- 信息结束: X分/2分 - [依据与说明]")
        parts.append("- 值长回应: X分/2分 - [依据与说明]")
        parts.append("- 整体规范: X分/2分 - [依据与说明]")
        parts.append("")
        parts.append("#### 总分: X分/10分")
        parts.append("")
        parts.append("#### 综合说明")
        parts.append("[对流程执行情况的总体评价，包括合规点与改进建议]")
        parts.append("")
        parts.append("请严格按以上格式输出，不要输出其他内容。")

        prompt = "\n".join(parts)
        prompt += """
## 重要输出指令（红线要求）
请务必在给出上述任何报告结构之前，先在 `<think>...</think>` 标签内输出你对该流程的详细分析、步步推导和思考推理过程。
格式必须为：
<think>
[这里是你的推理和思考过程，一步步分析每一项评分的扣分/给分依据，字数在 100-300 字左右]
</think>

[接下来是上述要求的报告内容]
"""
        return prompt

    def _build_supervision_prompt(self, flow_data: Dict) -> str:
        """构建监护制评估 prompt"""
        voice_events = flow_data.get("voice_events", [])
        tracker_events = flow_data.get("tracker_events", [])
        behavior_events = flow_data.get("behavior_events", [])

        # ============ 第一部分：完整的评估规程 prompt ============
        prompt = f"""你是核电站主控室监护制规程合规检测评审专家。请根据附带的关键事件（keymoment），评估该流程与监护制规程要求的合规性。

## 监护制规程要求（步骤顺序）
1. 流程启动：操作人喊出"请求监护" + 流程内有举手动作（红线规则：单独举手不启动流程，必须语音 + 举手）
2. 监护人到位：监护人移动至操作人身旁（跟踪事件"监护员已到位监护X回路"）
3. 指令复述：操作人读出9字码 → 监护人复述9字码
4. 执行命令：监护人下达"可以执行"命令（包含"执行"二字即可接受）
5. 核对确认：双方检查设备状态，喊出"核对"
6. 监护结束：监护人离开操作人（跟踪事件"监护员已离开监护X回路"）

## 评估维度（每项2分，满分10分）
| 维度 | 评估要点 | 评分标准 |
|------|---------|---------|
| 1. 流程启动 | 是否有"请求监护" + 流程内举手 | 2分=语音+举手；1分=仅语音；0分=无启动信号 |
| 2. 监护人到位 | 跟踪事件是否记录到位 | 2分=有到位记录；0分=无 |
| 3. 指令复述 | 操作人读9字码 + 监护人复述9字码 | 2分=双方都复述；1分=单方复述；0分=无复述 |
| 4. 执行命令 | 是否有含"执行"关键字的命令 | 2分=有执行命令；0分=无 |
| 5. 核对确认 | 是否有"核对" | 2分=有"核对"；0分=无 |

## 顺序一致性检查
请检查 keymoment 的时间顺序是否符合监护制要求：
- "请求监护"应在流程开始
- 监护人到位应在"请求监护"之后
- 9字码读出应在监护人到位之后
- "执行"应在9字码复述之后
- "核对"应在"执行"之后
- 监护人离开应在流程末尾
- 若顺序错乱，请在综合说明中指出

## 输出格式（严格按此格式输出）
### 监护制评估报告

#### 维度评分
- 流程启动: X分/2分 - [依据与说明]
- 监护人到位: X分/2分 - [依据与说明]
- 指令复述: X分/2分 - [依据与说明]
- 执行命令: X分/2分 - [依据与说明]
- 核对确认: X分/2分 - [依据与说明]

#### 顺序一致性: [符合/部分错乱/严重错乱]
- [若有错乱，列出具体错乱步骤]

#### 总分: X分/10分

#### 综合说明
[对流程执行情况的总体评价，包括合规点、顺序问题与改进建议]

## 流程信息
- 流程类型: 监护制
- {flow_data.get('_flow_id_desc', '')}
- 开始时间: {flow_data.get('flow_start_sec', 0)}s
- 结束时间: {flow_data.get('flow_end_sec', 0)}s
- 持续时间: {flow_data.get('flow_continue_sec', 0)}s
"""

        # ============ 第二部分：附带的 keymoment 数据 ============
        prompt += "\n## 附带的关键事件（keymoment）\n请根据以下 keymoment 进行评估：\n\n"

        prompt += "### 语音事件（按时间顺序）\n"
        for event in voice_events:
            prompt += f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}\n"
        if not voice_events:
            prompt += "- 无\n"

        prompt += "\n### 跟踪事件（监护到位/离开等）\n"
        for event in tracker_events:
            prompt += f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}\n"
        if not tracker_events:
            prompt += "- 无\n"

        prompt += "\n### 行为事件（举手等）\n"
        for event in behavior_events:
            prompt += f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}\n"
        if not behavior_events:
            prompt += "- 无\n"

        prompt += """
## 重要输出指令（红线要求）
请务必在给出上述任何报告结构之前，先在 `<think>...</think>` 标签内输出你对该流程的详细分析、步步推导和思考推理过程。
格式必须为：
<think>
[这里是你的推理和思考过程，一步步分析每一项评分的扣分/给分依据，字数在 100-300 字左右]
</think>

[接下来是上述要求的报告内容]
"""
        return prompt

    def _build_self_ticket_prompt(self, flow_data: Dict) -> str:
        """构建自唱票评估 prompt"""
        voice_events = flow_data.get("voice_events", [])
        behavior_events = flow_data.get("behavior_events", [])

        parts = []
        parts.append("你是核电站主控室自唱票规程合规检测评审专家。请根据以下流程数据，按五个评估维度进行评分。")
        parts.append("")
        parts.append("## 自唱票规程规则")
        parts.append("1. 流程启动：操作人右手点开操作控件")
        parts.append("2. 九字码读出：操作人读出控件9字码")
        parts.append("3. 九字码确认：")
        parts.append("   - 有程序计划：操作人左手指向程序指令（关键特征），核对9字码与程序一致")
        parts.append("   - 无程序计划：操作人确认读出的9字码与想要操作的设备一致")
        parts.append("4. 设备操作：操作人执行设备操作")
        parts.append("5. 流程结束：操作完成")
        parts.append("")
        parts.append("## 流程信息")
        parts.append(f"- 流程类型: 自唱票")
        if flow_data.get('_flow_id_desc'):
            parts.append(f"- {flow_data.get('_flow_id_desc')}")
        parts.append(f"- 开始时间: {flow_data.get('flow_start_sec', 0)}s")
        parts.append(f"- 结束时间: {flow_data.get('flow_end_sec', 0)}s")
        parts.append(f"- 持续时间: {flow_data.get('flow_continue_sec', 0)}s")
        parts.append(f"- 设备代码: {flow_data.get('device_code', '未知')}")
        parts.append("")
        parts.append("## 语音事件（按时间排序）")
        for event in voice_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        if not voice_events:
            parts.append("- 无")
        parts.append("")
        parts.append("## 行为事件（手指屏幕、操作手势等）")
        for event in behavior_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        if not behavior_events:
            parts.append("- 无")
        parts.append("")
        parts.append("## 评估维度（每项2分，满分10分）")
        parts.append("| 维度 | 评估要点 | 评分标准 |")
        parts.append("|------|---------|---------|")
        parts.append("| 1. 流程启动 | 操作人是否点开操作控件 | 2分=有点开动作；0分=无 |")
        parts.append("| 2. 九字码读出 | 操作人是否读出9字码 | 2分=读出9字码；0分=无 |")
        parts.append("| 3. 九字码确认 | 操作人是否确认9字码一致 | 2分=有确认动作；1分=部分确认；0分=无确认 |")
        parts.append("| 4. 设备操作 | 操作人是否执行设备操作 | 2分=有操作；0分=无 |")
        parts.append("| 5. 整体规范 | 流程完整度与时间合理性 | 2分=完整规范；1分=基本规范；0分=不规范 |")
        parts.append("")
        parts.append("## 输出格式")
        parts.append("请按以下格式输出评估报告：")
        parts.append("")
        parts.append("### 自唱票评估报告")
        parts.append("")
        parts.append("#### 维度评分")
        parts.append("- 流程启动: X分/2分 - [依据与说明]")
        parts.append("- 九字码读出: X分/2分 - [依据与说明]")
        parts.append("- 九字码确认: X分/2分 - [依据与说明]")
        parts.append("- 设备操作: X分/2分 - [依据与说明]")
        parts.append("- 整体规范: X分/2分 - [依据与说明]")
        parts.append("")
        parts.append("#### 总分: X分/10分")
        parts.append("")
        parts.append("#### 综合说明")
        parts.append("[对流程执行情况的总体评价，包括合规点与改进建议]")
        parts.append("")
        parts.append("请严格按以上格式输出，不要输出其他内容。")

        prompt = "\n".join(parts)
        prompt += """
## 重要输出指令（红线要求）
请务必在给出上述任何报告结构之前，先在 `<think>...</think>` 标签内输出你对该流程的详细分析、步步推导和思考推理过程。
格式必须为：
<think>
[这里是你的推理和思考过程，一步步分析每一项评分的扣分/给分依据，字数在 100-300 字左右]
</think>

[接下来是上述要求的报告内容]
"""
        return prompt

    def _extract_score(self, report_text: str) -> int:
        """从报告文本中提取评分"""
        import re

        # 清除 <think>...</think> 思考块的干扰，防止匹配到大模型在思考推理过程中演算的各项中间分值
        clean_text = re.sub(r"<think>.*?</think>", "", report_text, flags=re.DOTALL)

        patterns = [
            r"总分[：:]\s*(\d+)",
            r"得分[：:]\s*(\d+)",
            r"(\d+)/10",
            r"评分[：:]\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                return int(match.group(1))

        logger.warning("无法从报告中提取评分，默认返回 5 分")
        return 5
