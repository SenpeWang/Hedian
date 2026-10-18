// 通知数据(状态量+事件流双语义):
// 状态量(人数/凝视): 后端 globalSec 对齐后实时推送, 前端收到即实时显示(推理侧视角, 不被播放时刻卡住)
// 事件流(流程开始/结束): 与画面严格同步(sec ≤ 播放时刻, 播放侧视角)
import { ref, computed, type Ref } from 'vue'
import type {
  BehaviorEvent,
  FlowEvent,
  FlowType,
  GazeState,
  NotifyEvent,
  PeopleState
} from '../types'
import { createLogger } from '../media/logger'

const log = createLogger('notify')

// 行为检测事件 → [颜色, 中文名]: 与流程事件同属事件流(按播放时刻对齐)
const behaviorTypeMap: Record<string, [string, string]> = {
  HAND_RAISED: ['#ffaa00', '举手'],
  FINGER_SCREEN: ['#00d4ff', '手指指向屏幕'],
  FINGER_FILE: ['#00ffcc', '手指指向文件']
}

const flowTypeMap: Record<FlowType, [string, string]> = {
  supervision: ['#00d4ff', '监护制'],
  self_ticket: ['#ffaa00', '自唱票'],
  info_notice: ['#00ffcc', '信息通报'],
  normal_comm: ['#00ff88', '普通沟通']
}

/**
 * 通知数据(状态量+事件流双语义):
 * 状态量(人数/凝视): 后端 globalSec 对齐后实时推送, 前端收到即实时显示(推理侧视角, 不被播放时刻卡住);
 * 事件流(流程开始/结束): 与画面严格同步(sec ≤ 播放时刻, 播放侧视角).
 * @param _currentPlaybackSec 主时钟引用(事件流同步过滤用, 预留参数)
 * @returns 状态量 refs / 同步后的 `flowEvents`, 以及 `addTracking`/`addGaze`/`addFlow`
 *   三个消息入口与 `reset`
 */
export function useNotify(_currentPlaybackSec: Ref<number>) {
  // 状态量: 实时显示后端对齐推送的最新值(常量内存, 无需时序池/驱逐)
  const people = ref<PeopleState>({ count: '--', alert: '就绪', alertColor: '#8899aa' })
  const gaze = ref<GazeState>({ hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 })
  const rawFlowEvents = ref<FlowEvent[]>([])
  const rawBehaviorEvents = ref<BehaviorEvent[]>([])

  function addTracking(d: unknown): void {
    const src = d as {
      tag?: string
      localSec?: number
      data?: { count?: number; state_alert?: string }
    }
    if (src.tag !== 'PEOPLE_COUNT_UPDATE') return
    const dt = src.data || {}
    people.value = {
      count: dt.count ?? '--',
      alert: dt.state_alert ? '⚠️ ' + dt.state_alert : '✅ 当前人数正常',
      alertColor: dt.state_alert ? '#ff4d4d' : '#00ff88'
    }
    log.debug(`人数更新 localSec=${src.localSec} count=${dt.count}`)
  }

  function addGaze(items: unknown): void {
    const arr = items as { localSec?: number; data?: Record<string, unknown> }[] | undefined
    if (!arr || !arr[0]?.data) return
    const dt = arr[0].data
    gaze.value = {
      hasHeads: !!dt.has_heads,
      headsCount: (dt.heads_count as number) || 0,
      anyInRoi: !!dt.any_in_roi,
      awayDuration: (dt.away_duration as number) || 0
    }
  }

  function addFlow(d: unknown, isStart: boolean): void {
    const src = d as {
      localSec?: number
      sec?: number
      flowType?: FlowType
      data?: Record<string, unknown>
    }
    const dt = src.data || {}
    const sec = Number(
      isStart
        ? dt.flow_start_sec || src.localSec || src.sec || 0
        : dt.flow_end_sec || src.localSec || src.sec || 0
    )
    const flowType = ((dt.flow_type as FlowType) || src.flowType || 'supervision') as FlowType
    const [color, name] = flowTypeMap[flowType]
    if (
      rawFlowEvents.value.find(
        (e) => e.flowType === flowType && Math.abs(e.sec - sec) < 0.1 && e.isStart === isStart
      )
    ) {
      return
    }
    const item: FlowEvent = {
      sec: Number(sec.toFixed(2)),
      flowType,
      name,
      color,
      isStart,
      flowId: Number(dt.flow_id || 0)
    }
    // 有序插入，而不是无脑 push。
    // 原因：App.vue 是「先把整批 flow_start 全部塞进列表，再塞 flow_end」，
    // 批次内的真实先后被丢掉；于是同一秒里"开始"会排到"结束"之前，
    // 看起来像 0 秒流程 / 两个流程并存。
    // 排序键：sec 升序 → flow_id 升序 → 同一 flow 内"开始"在"结束"之前。
    //   · 被新的普通沟通顶掉：同一秒「旧流程结束(小 flow_id)」→「新流程开始(大 flow_id)」
    //   · 同句应答：同一秒、同一个 flow_id 的「开始」→「结束」
    const keyOf = (e: FlowEvent): [number, number, number] => [
      e.sec,
      e.flowId ?? 0,
      e.isStart ? 0 : 1
    ]
    const keyNew = keyOf(item)
    const idx = rawFlowEvents.value.findIndex((e) => {
      const k = keyOf(e)
      if (k[0] !== keyNew[0]) return k[0] > keyNew[0]
      if (k[1] !== keyNew[1]) return k[1] > keyNew[1]
      return k[2] > keyNew[2]
    })
    if (idx < 0) rawFlowEvents.value.push(item)
    else rawFlowEvents.value.splice(idx, 0, item)
  }

  // 行为检测事件: 去重 + 按时刻有序插入(同流程事件)
  function addBehavior(d: unknown): void {
    const src = d as { localSec?: number; tag?: string }
    const sec = Number(src.localSec ?? 0)
    const tag = String(src.tag || '')
    const [color, name] = behaviorTypeMap[tag] ?? ['#8899aa', tag || '行为事件']
    if (rawBehaviorEvents.value.find((e) => e.tag === tag && Math.abs(e.sec - sec) < 0.1)) {
      return
    }
    const item: BehaviorEvent = { sec: Number(sec.toFixed(2)), tag, name, color }
    const idx = rawBehaviorEvents.value.findIndex((e) => e.sec > item.sec)
    if (idx < 0) rawBehaviorEvents.value.push(item)
    else rawBehaviorEvents.value.splice(idx, 0, item)
  }

  // 事件流: 严格同步, 与画面同一时刻(无提前量)
  const flowEvents = computed(() =>
    rawFlowEvents.value.filter((e) => e.sec <= _currentPlaybackSec.value)
  )

  // 行为事件: 与流程事件同一过滤语义(sec ≤ 播放时刻), 保证与画面对齐
  const behaviorEvents = computed(() =>
    rawBehaviorEvents.value.filter((e) => e.sec <= _currentPlaybackSec.value)
  )

  // 通知栏统一事件流: 流程事件 + 行为检测事件, 合并后按时刻排序
  const notifyEvents = computed<NotifyEvent[]>(() =>
    [
      ...flowEvents.value.map((e) => ({
        sec: e.sec,
        text: `${e.name}${e.isStart ? '开始' : '结束'}`,
        color: e.color,
        key: `flow-${e.flowType}-${e.sec}-${e.isStart ? 1 : 0}-${e.flowId ?? 0}`
      })),
      ...behaviorEvents.value.map((e) => ({
        sec: e.sec,
        text: e.name,
        color: e.color,
        key: `behavior-${e.tag}-${e.sec}`
      }))
    ].sort((a, b) => a.sec - b.sec || a.key.localeCompare(b.key))
  )

  function reset(): void {
    people.value = { count: '--', alert: '就绪', alertColor: '#8899aa' }
    gaze.value = { hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 }
    rawFlowEvents.value = []
    rawBehaviorEvents.value = []
  }

  return {
    people,
    gaze,
    flowEvents,
    notifyEvents,
    addTracking,
    addGaze,
    addFlow,
    addBehavior,
    reset
  }
}
