#!/usr/bin/env python3
"""
核电站监护制合规检测 — 大模型评估模块
Qwen2.5-1.5B-Instruct 本地推理
"""
import os, json, re, argparse

# ── 强制离线模式，防止 transformers 尝试访问 HuggingFace ──
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── 模型路径 ──
_QWEN_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.abspath(os.path.join(_QWEN_DIR, "..", "..", "..", "Hedian_data"))
DEFAULT_MODEL = os.path.join(_DATA_ROOT, "Qwen_model")

# ── System Prompt（精简，不包含输出模板避免小模型照抄） ──
SYSTEM_PROMPT = """你是核电站监护制合规检测评审专家。你需要根据数据打分。

评分标准（总分100）：
- 三段式沟通（70分）：发令含设备编号20分，复述关键内容30分，确认闭环20分
- 监护制度（30分）：监护请求10分，监护到位10分，监护持续性10分

评级：A(90-100) B(80-89) C(60-79) D(<60)

请严格按数据打分，给出具体分数和理由。"""


def _build_user_prompt(voice_events, tracking_events, mini_reports=None, kf_files=None):
    """组装 User Prompt — 预处理数据摘要，引导模型逐项评分"""
    parts = []

    # ── 数据摘要（减少 token，帮小模型抓重点） ──
    # 统计设备编号出现次数
    import re as _re
    dev_pattern = _re.compile(r'(1ES\w+|T1RPA\w+|LCO\w+|RPA\w+|SM3)')
    cmds_with_dev = 0
    cmds_total = 0
    confirm_count = 0
    has_supervision_req = False
    for ev in voice_events:
        intent = ev.get("intent", "")
        text = ev.get("text", "")
        if intent == "操作指令":
            cmds_total += 1
            if dev_pattern.search(text):
                cmds_with_dev += 1
        if intent in ("确认", "监护确认"):
            confirm_count += 1
        if intent == "监护请求" or "请求监护" in text:
            has_supervision_req = True

    # 统计监护状态
    sup_close = 0
    sup_total = 0
    hand_raised = False
    for ev in tracking_events:
        e = ev.get("event", "")
        if e == "SUPERVISOR_STATUS":
            sup_total += 1
            if ev.get("status") == "监护中":
                sup_close += 1
        if e in ("HAND_RAISE_SUPERVISION", "HAND_RAISE_CHECK"):
            if ev.get("raised", True):
                hand_raised = True

    # 分段评价摘要
    if mini_reports:
        avg_score = sum(m.get("score", 0) for m in mini_reports) / len(mini_reports) if mini_reports else 0
        mr_text = "\n".join(
            f"  流程{i+1} [{m.get('time','?')}] {m.get('device','?')} {m.get('score','?')}/10 — {m.get('detail','')}"
            for i, m in enumerate(mini_reports)
        )
        parts.append(f"分段评价（{len(mini_reports)}个流程，均分{avg_score:.1f}）：\n{mr_text}")

    # 数据摘要
    parts.append(f"""数据摘要：
- 操作指令{cmds_total}条，其中{cmds_with_dev}条含设备编号
- 确认语句{confirm_count}条
- 监护请求：{'有' if has_supervision_req else '无'}
- 举手动作：{'检测到' if hand_raised else '未检测到'}
- 监护状态采样{sup_total}次，到位{sup_close}次（{'%.0f' % (sup_close/sup_total*100) if sup_total else 0}%）""")

    # 关键语音（只取操作指令和监护相关）
    key_voice = [ev for ev in voice_events if ev.get("intent") in ("操作指令", "监护请求", "确认", "监护确认", "操作结束", "实验结束")]
    voice_summary = "\n".join(
        f"  [{ev.get('time_sec',0):.0f}s] [{ev.get('intent','')}] {ev.get('text','')[:80]}"
        for ev in key_voice[:20]
    )
    parts.append(f"关键语音记录：\n{voice_summary}")
    
    # 引导式评分（严厉规范小模型必须自行输出数字）
    parts.append(f"""请根据以上提供的事实数据，对各项合规指标进行自主打分，并给出评价理由。
    
【严厉指令】：
1. 三段式沟通（每项满分分别为20分/30分/20分）和监护制度（每项满分均为10分）。
2. 你必须在【得分：?】的问号处，填写一个准确的阿拉伯数字评分（例如【得分：18】分）。绝不可跳过、用汉字代替或只写评语！
3. 请严格按照给定的"评分明细"格式续写。

## 评分明细

### 一、三段式沟通（满分70分）
1. 发令完整性（满分20分）：【得分：?】分 — {cmds_with_dev}/{cmds_total}条含设备编号，（你的评价）
2. 复述准确性（满分30分）：【得分：?】分 — （根据语音记录判断，你的评价）
3. 确认闭环（满分20分）：【得分：?】分 — 共{confirm_count}次确认，（你的评价）

### 二、监护制度（满分30分）
1. 监护请求（满分10分）：【得分：?】分 — {'有语音请求且有举手' if has_supervision_req and hand_raised else '缺少请求或未完整举手'}，（你的评价）
2. 监护到位（满分10分）：【得分：?】分 — 到位率{'%.0f' % (sup_close/sup_total*100) if sup_total else 0}%，（你的评价）
3. 监护持续性（满分10分）：【得分：?】分 — （你的评价）

## 总分：?/100  评级：?

## 问题清单
- （列出扣分项总结）

## 总体评价
（50字以内总结）""")

    return "\n\n".join(parts)


def generate_report(voice_events, tracking_events, mini_reports=None,
                    kf_files=None, model_path=None, progress_cb=None):
    """
    生成合规评估报告（本地 Qwen 推理，无回退）

    Returns: (report_text, total_score, grade)
    """
    user_prompt = _build_user_prompt(voice_events, tracking_events, mini_reports, kf_files)

    if progress_cb:
        progress_cb("正在加载 Qwen2.5 模型…")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    mp = model_path or DEFAULT_MODEL
    print(f"[Qwen] 加载模型: {mp}")

    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, torch_dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
    )

    if progress_cb:
        progress_cb("大模型推理中，请稍候…")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    report = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # 提取总分
    m = re.search(r"总分[：:]\s*(\d+)", report)
    total = int(m.group(1)) if m else 0
    m2 = re.search(r"评级[：:]\s*([ABCD])", report)
    grade = m2.group(1) if m2 else ("A" if total>=90 else "B" if total>=80 else "C" if total>=60 else "D")

    if progress_cb:
        progress_cb(f"评估完成: {total}分 评级{grade}")

    return report, total, grade


# ── 命令行入口 ──
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大模型合规评估")
    parser.add_argument("--voice", required=True, help="voice_events.json 路径")
    parser.add_argument("--tracking", required=True, help="tracking_events.json 路径")
    parser.add_argument("--mini", default=None, help="mini_reports.json 路径")
    parser.add_argument("--model", default=None, help="Qwen 模型路径")
    parser.add_argument("--output", default="compliance_report.md", help="输出报告路径")
    args = parser.parse_args()

    with open(args.voice, encoding="utf-8") as f:
        ve = json.load(f)
    with open(args.tracking, encoding="utf-8") as f:
        te = json.load(f)
    mr = None
    if args.mini:
        with open(args.mini, encoding="utf-8") as f:
            mr = json.load(f)

    report, total, grade = generate_report(ve, te, mr, model_path=args.model)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {args.output} (总分: {total}, 评级: {grade})")
