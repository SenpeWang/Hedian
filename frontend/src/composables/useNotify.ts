// 通知数据(状态量+事件流双语义):
// 状态量(人数/凝视): 后端 globalSec 对齐后实时推送, 前端收到即实时显示(推理侧视角, 不被播放时刻卡住)
// 事件流(流程开始/结束): 与画面严格同步(sec ≤ 播放时刻, 播放侧视角)
import { ref, computed, type Ref } from 'vue'
import type { FlowEvent, FlowType, GazeState, PeopleState } from '../types'
import { createLogger } from '../media/logger'

const log = createLogger('notify')

const flowTypeMap: Record<FlowType, [string, string]> = {
  supervision: ['#00d4ff', '监护制'],
  self_ticket: ['#ffaa00', '自唱票'],
  info_notice: ['#00ffcc', '信息通报'],
}

export function useNotify(_currentPlaybackSec: Ref<number>) {
  // 状态量: 实时显示后端对齐推送的最新值(常量内存, 无需时序池/驱逐)
  const people = ref<PeopleState>({ count: '--', alert: '就绪', alertColor: '#8899aa' })
  const gaze = ref<GazeState>({ hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 })
  const rawFlowEvents = ref<FlowEvent[]>([])

  function addTracking(d: unknown): void {
    const src = d as { tag?: string; localSec?: number; data?: { count?: number; state_alert?: string } }
    if (src.tag !== 'PEOPLE_COUNT_UPDATE') return
    const dt = src.data || {}
    people.value = {
      count: dt.count ?? '--',
      alert: dt.state_alert ? '⚠️ ' + dt.state_alert : '✅ 当前人数正常',
      alertColor: dt.state_alert ? '#ff4d4d' : '#00ff88',
    }
    log.debug(`人数更新 localSec=${src.localSec} count=${dt.count}`)
  }

  function addGaze(items: unknown): void {
    const arr = items as { localSec?: number; data?: Record<string, unknown> }[] | undefined
    if (!arr || !arr[0]?.data) return
    const dt = arr[0].data
    gaze.value = {
      hasHeads: !!dt.has_heads, headsCount: (dt.heads_count as number) || 0,
      anyInRoi: !!dt.any_in_roi, awayDuration: (dt.away_duration as number) || 0,
    }
  }

  function addFlow(d: unknown, isStart: boolean): void {
    const src = d as { localSec?: number; sec?: number; flowType?: FlowType; data?: Record<string, unknown> }
    const dt = src.data || {}
    const sec = Number(isStart
      ? (dt.flow_start_sec || src.localSec || src.sec || 0)
      : (dt.flow_end_sec || src.localSec || src.sec || 0))
    const flowType = ((dt.flow_type as FlowType) || src.flowType || 'supervision') as FlowType
    const [color, name] = flowTypeMap[flowType]
    if (!rawFlowEvents.value.find(e => e.flowType === flowType && Math.abs(e.sec - sec) < 0.1 && e.isStart === isStart)) {
      rawFlowEvents.value.push({ sec: Number(sec.toFixed(2)), flowType, name, color, isStart })
    }
  }

  // 事件流: 严格同步, 与画面同一时刻(无提前量)
  const flowEvents = computed(() =>
    rawFlowEvents.value.filter(e => e.sec <= _currentPlaybackSec.value)
  )

  function reset(): void {
    people.value = { count: '--', alert: '就绪', alertColor: '#8899aa' }
    gaze.value = { hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 }
    rawFlowEvents.value = []
  }

  return { people, gaze, flowEvents, addTracking, addGaze, addFlow, reset }
}
