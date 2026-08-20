import { ref, watch, onBeforeUnmount, type Ref } from 'vue'
import type { SegCard } from '../types'

// 慢速逐字打字机: setInterval 60ms/字(~16字/秒), 模拟大模型流式输出
// streamBuffer 按 chunk 累积(后端到达), shownLen 逐字追赶; 追完 + reportText 到达 → 切 streaming=false
// 停止条件: 所有卡片转完成态才停(防 chunk 续到时 timer 已停致卡死)
function parseReportContent(text: string): { think: string; report: string } {
  if (!text) return { think: '', report: '' }
  const thinkMatch = text.match(/<think>([\s\S]*?)(?:<\/think>|$)/i)
  const think = thinkMatch ? thinkMatch[1].trim() : ''
  const report = text.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim()
  return { think, report }
}

export function useTypewriter(cards: Ref<SegCard[]>, step = 60) {
  const shownLen = ref<Record<string, number>>({})
  let timerId: ReturnType<typeof setInterval> | null = null

  function tick() {
    for (const card of cards.value) {
      if (!card.streaming) continue
      const full = card.streamBuffer || card.reportText || ''
      const cur = shownLen.value[card.flowId] || 0
      if (cur < full.length) {
        shownLen.value[card.flowId] = Math.min(cur + 1, full.length)
      } else if (card.reportText) {
        // 追赶到末尾 + 终态(segment_report)到达 → 切完成态(显分数)
        card.streaming = false
      }
    }
    // 所有卡片转完成态才停(防 chunk 续到时 timer 已停致卡死; 60ms 空转开销可忽略)
    if (!cards.value.some(c => c.streaming) && timerId !== null) { clearInterval(timerId); timerId = null }
  }

  // 有 streaming 卡片即启动, 一直跑到全部完成
  watch(() => cards.value.some(c => c.streaming), (has) => {
    if (has && timerId === null) timerId = setInterval(tick, step)
  }, { immediate: true })

  onBeforeUnmount(() => { if (timerId !== null) clearInterval(timerId) })

  // 截 streamBuffer 按 shownLen, 再 parse 出对应 field(think+report 都逐字)
  function shownText(card: SegCard, field: 'think' | 'report'): string {
    const raw = card.streamBuffer || card.reportText || ''
    if (!raw) return ''
    if (!card.streaming) return parseReportContent(raw)[field]
    const len = shownLen.value[card.flowId] || 0
    return parseReportContent(raw.slice(0, Math.min(len, raw.length)))[field]
  }

  return { shownLen, shownText }
}
