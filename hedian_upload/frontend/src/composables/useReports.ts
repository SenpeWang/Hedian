// 评估报告卡片: 流式 chunk 累积 + 完成态 + 计数; 完成态释放全文副本(RetentionCenter)
// 评估走直推通道(不参与对齐), 前端播到流程结束点附近由卡片渲染时序承接(语义同现状)
import { ref, computed, watch } from 'vue'
import type { SegCard, FlowType } from '../types'
import { retentionCenter } from '../media/retention'
import { isStreamMsg, isTerminalMsg } from '../media/protocol'
import { toCachedCard, fromCachedCard, type CachedCard } from '../media/panelCache'

export function useReports() {
  const segCards = ref<SegCard[]>([])
  const segScores = ref<number[]>([])
  const supN = ref(0)
  const noticeN = ref(0)
  const completedFlows = new Set<string>()

  function handleReportEvent(msg: unknown): void {
    const m = msg as { source?: string; type?: string; data?: Record<string, unknown>; flow_id?: string; text?: string }
    const data = (m.data || {}) as Record<string, unknown>
    const flowId = String(data.flow_id || m.flow_id || '')
    // flow_type 仅在流式首帧到达前可能缺失；缺失时留空，待后续消息回填，
    // 不再兜底成 supervision（否则"信息通报"卡片会被误标为"监护制"）
    const flowType = ((data.flow_type as FlowType) || (m as { flow_type?: FlowType }).flow_type || '') as FlowType | ''
    // 卡片主键必须带制度类型：各制度的 flow_id 都从 1 独立自增，
    // info_notice#1 与 supervision#1 的 flow_id 都是 "1"。只用 flow_id 做键时，
    // 后到的监护制#1 会把信息通报#1 的卡片整个顶掉（标签/时长/正文全被改写，
    // 表现为"第一个是监护制[206.84s]"，且信息通报#1 凭空消失）
    const flowKey = `${flowType || '?'}:${flowId}`
    const reportText = (data.report_text as string) || (data.text as string) || m.text || ''
    const score = Number(data.score !== undefined ? data.score : 0)
    // 后端 segment_report 推的是 flow_continue_sec（见 flow_evaluation_manager），
    // 此处必须优先读它，否则卡片头时长恒为 0 显示成 [0s]
    const continueSec =
      (data.flow_continue_sec as number) ||
      (data.continue_sec as number) ||
      (data.duration as number) ||
      0
    // 流程真实时间区间：评估按「流程结束」排队，卡片到达顺序 ≠ 流程发生顺序，
    // 排序与卡片头区间都依赖它（缺失时保持到达顺序，不硬塞 0 把所有卡挤到最前）
    const startSec = data.flow_start_sec === undefined || data.flow_start_sec === null
      ? null
      : Number(data.flow_start_sec)
    const endSec = data.flow_end_sec === undefined || data.flow_end_sec === null
      ? null
      : Number(data.flow_end_sec)
    if (!flowId) return
    let card = segCards.value.find(c => c.flowKey === flowKey)
    // 兜底: 流式首帧若未带 flow_type, 会先按 `?:id` 建占位卡; 类型到达后认领它并升格主键
    if (!card && flowType) {
      const placeholder = segCards.value.find(c => !c.flowType && c.flowId === flowId)
      if (placeholder) {
        placeholder.flowType = flowType
        placeholder.flowKey = flowKey
        card = placeholder
      }
    }
    // token 到达驱动: segment_report_stream 的 data.chunk 按到达累积进 streamBuffer
    const chunk = (data.chunk as string) || ''
    const isStream = isStreamMsg(m)
    if (!card) {
      // 流式态新建: reportText 留空(终态才写), 避免 streamBuffer 与 reportText 混淆
      card = { flowKey, flowId, flowType, score, reportText: isStream ? '' : reportText, continueSec, startSec, endSec, collapsed: false, streamBuffer: chunk || reportText, streaming: isStream }
      segCards.value.push(card)
    } else {
      if (score > 0) card.score = score
      if (continueSec) card.continueSec = continueSec
      if (startSec !== null && Number.isFinite(startSec)) card.startSec = startSec
      if (endSec !== null && Number.isFinite(endSec)) card.endSec = endSec
      // 类型只允许"空 → 有"的回填, 绝不覆盖已确定的类型(否则卡片会串台改标)
      if (flowType && !card.flowType) card.flowType = flowType
      // 流式 chunk 累积(按 token 到达逐字); 终态用完整 report_text 覆盖
      if (chunk && isStream) {
        card.streamBuffer += chunk
      }
      if (reportText && isTerminalMsg(m)) {
        card.reportText = reportText
        // 不覆盖 streamBuffer、不设 streaming=false: 让 typewriter 自然逐字到 streamBuffer 末尾再停
      }
    }
    if (isTerminalMsg(m) && !completedFlows.has(flowKey)) {
      completedFlows.add(flowKey)
      if (score > 0) segScores.value.push(score)
      // 计数用回填后的卡片类型，避免流式首帧缺 flow_type 时漏计/错计
      const finalType = flowType || card.flowType
      if (finalType === 'supervision') supN.value++
      else if (finalType === 'info_notice') noticeN.value++
    }
  }

  // 完成态翻转(typewriter 追完) → 释放全文副本(显示 fallback reportText)
  watch(() => segCards.value.map(c => `${c.flowKey}:${c.streaming}`).join('|'), () => {
    for (const c of segCards.value) {
      if (!c.streaming && c.streamBuffer && c.reportText) retentionCenter.releaseReportCopy(c)
    }
  })

  // 展示顺序：按流程真实开始时刻升序。评估报告串行产出，到达顺序 = 流程结束顺序，
  // 会把晚开始的流程排到早开始的流程之前；排序后卡片顺序 = 视频里发生的顺序。
  // 缺 startSec 的卡（旧版后端/流式首帧）保持到达顺序排在有时刻的卡之后。
  const orderedCards = computed(() =>
    segCards.value.slice().sort((a, b) => {
      const sa = Number(a.startSec)
      const sb = Number(b.startSec)
      const aOk = a.startSec !== null && a.startSec !== undefined && Number.isFinite(sa)
      const bOk = b.startSec !== null && b.startSec !== undefined && Number.isFinite(sb)
      if (aOk && bOk) return sa - sb
      if (aOk) return -1
      if (bOk) return 1
      return 0
    })
  )

  // 已完成的卡片(流式中的不缓存): 供刷新恢复落 localStorage
  const finishedCards = computed(() => segCards.value.filter(c => !c.streaming).map(toCachedCard))

  const totalScore = computed(() => segScores.value.reduce((a, b) => a + b, 0))
  const avgScore = computed(() => segScores.value.length > 0 ? (totalScore.value / segScores.value.length).toFixed(1) : '-')

  // SegCard 展示态收口: collapsed 折叠由本 store 持有修改权, 组件 emit 调用(不直改 prop)
  function toggleCard(flowKey: string): void {
    const card = segCards.value.find(c => c.flowKey === flowKey)
    if (card) card.collapsed = !card.collapsed
  }

  // 刷新恢复: 回填上轮已完成的报告卡片与计数/分数(流式态卡片不缓存,
  // 恢复后一律为完成态, 打字机直接显示全文)
  function restore(
    items: CachedCard[],
    counts?: { sup?: number; ticket?: number; notice?: number },
    scores?: number[],
  ): void {
    for (const item of items || []) {
      segCards.value.push(fromCachedCard(item))
      completedFlows.add(item.flowKey)
    }
    supN.value = counts?.sup || 0
    noticeN.value = counts?.notice || 0
    if (scores && scores.length) segScores.value = scores.slice()
  }

  function reset(): void {
    segCards.value = []; segScores.value = []
    supN.value = 0; noticeN.value = 0
    completedFlows.clear()
  }

  return { segCards, orderedCards, finishedCards, segScores, supN, noticeN, totalScore, avgScore, toggleCard, handleReportEvent, restore, reset }
}
