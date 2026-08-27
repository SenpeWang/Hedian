// 字幕事件池: voice 数据解析 + 严格同步取数(sec ≤ 主时钟, 全量保留=完整对话记录)
import { reactive, computed, type Ref } from 'vue'
import type { VoiceEntry } from '../types'

export function useTranscript(currentPlaybackSec: Ref<number>) {
  const rawVoiceMap = reactive<Record<number, VoiceEntry>>({})

  function addVoice(d: unknown): void {
    const src = d as { localSec?: number; data?: { sec?: number; text?: string; keys?: string[] } }
    const dt = src.data || {}
    const sec = src.localSec || dt.sec || 0
    if (!rawVoiceMap[sec]) rawVoiceMap[sec] = { sec, text: dt.text || '', keys: dt.keys || [] }
    else if (dt.text) { rawVoiceMap[sec].text = dt.text; if (dt.keys) rawVoiceMap[sec].keys = dt.keys }
  }

  // 严格同步: 与画面同一时刻, 无提前量
  const voiceEntries = computed(() =>
    Object.values(rawVoiceMap).filter(v => v.sec <= currentPlaybackSec.value).sort((a, b) => a.sec - b.sec)
  )

  function reset(): void {
    for (const k in rawVoiceMap) delete rawVoiceMap[Number(k)]
  }

  return { voiceEntries, addVoice, reset }
}
