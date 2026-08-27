// 评估报告卡片: 流式 chunk 累积 + 完成态 + 计数; 完成态释放全文副本(RetentionCenter)
// 评估走直推通道(不参与对齐), 前端播到流程结束点附近由卡片渲染时序承接(语义同现状)
import { ref, computed, watch } from 'vue'
import type { SegCard, FlowType } from '../types'
import { retentionCenter } from '../media/retention'
import { isStreamMsg, isTerminalMsg } from '../media/protocol'

export function useReports() {
  const segCards = ref<SegCard[]>([])
  const segScores = ref<number[]>([])
  const supN = ref(0)
  const ticketN = ref(0)
  const noticeN = ref(0)
  const completedFlows = new Set<string>()

  function handleReportEvent(msg: unknown): void {
    const m = msg as { source?: string; type?: string; data?: Record<string, unknown>; flow_id?: string; text?: string }
    const data = (m.data || {}) as Record<string, unknown>
    const flowId = String(data.flow_id || m.flow_id || '')
    const flowType = ((data.flow_type as FlowType) || (m as { flow_type?: FlowType }).flow_type || 'supervision') as FlowType
    const reportText = (data.report_text as string) || (data.text as string) || m.text || ''
    const score = Number(data.score !== undefined ? data.score : 0)
    const continueSec = (data.continue_sec as number) || (data.duration as number) || 0
    if (!flowId) return
    let card = segCards.value.find(c => c.flowId === flowId)
    // token 到达驱动: segment_report_stream 的 data.chunk 按到达累积进 streamBuffer
    const chunk = (data.chunk as string) || ''
    const isStream = isStreamMsg(m)
    if (!card) {
      // 流式态新建: reportText 留空(终态才写), 避免 streamBuffer 与 reportText 混淆
      card = { flowId, flowType, score, reportText: isStream ? '' : reportText, continueSec, collapsed: false, streamBuffer: chunk || reportText, streaming: isStream }
      segCards.value.push(card)
    } else {
      if (score > 0) card.score = score
      if (continueSec) card.continueSec = continueSec
      // 流式 chunk 累积(按 token 到达逐字); 终态用完整 report_text 覆盖
      if (chunk && isStream) {
        card.streamBuffer += chunk
      }
      if (reportText && isTerminalMsg(m)) {
        card.reportText = reportText
        // 不覆盖 streamBuffer、不设 streaming=false: 让 typewriter 自然逐字到 streamBuffer 末尾再停
      }
    }
    if (isTerminalMsg(m) && !completedFlows.has(flowId)) {
      completedFlows.add(flowId)
      if (score > 0) segScores.value.push(score)
      if (flowType === 'supervision') supN.value++
      else if (flowType === 'self_ticket') ticketN.value++
      else if (flowType === 'info_notice') noticeN.value++
    }
  }

  // 完成态翻转(typewriter 追完) → 释放全文副本(显示 fallback reportText)
  watch(() => segCards.value.map(c => `${c.flowId}:${c.streaming}`).join('|'), () => {
    for (const c of segCards.value) {
      if (!c.streaming && c.streamBuffer && c.reportText) retentionCenter.releaseReportCopy(c)
    }
  })

  const totalScore = computed(() => segScores.value.reduce((a, b) => a + b, 0))
  const avgScore = computed(() => segScores.value.length > 0 ? (totalScore.value / segScores.value.length).toFixed(1) : '-')

  // SegCard 展示态收口: collapsed 折叠由本 store 持有修改权, 组件 emit 调用(不直改 prop)
  function toggleCard(flowId: string): void {
    const card = segCards.value.find(c => c.flowId === flowId)
    if (card) card.collapsed = !card.collapsed
  }

  function reset(): void {
    segCards.value = []; segScores.value = []
    supN.value = 0; ticketN.value = 0; noticeN.value = 0
    completedFlows.clear()
  }

  return { segCards, supN, ticketN, noticeN, totalScore, avgScore, toggleCard, handleReportEvent, reset }
}
