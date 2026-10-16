// 通知数据(状态量+事件流双语义):
// 状态量(人数/凝视): 后端 globalSec 对齐后实时推送, 前端收到即实时显示(推理侧视角, 不被播放时刻卡住)
// 事件流(流程开始/结束): 与画面严格同步(sec ≤ 播放时刻, 播放侧视角)
import { ref, computed, type Ref } from 'vue'
import type { FlowEvent, FlowKind, FlowType, GazeState, PeopleState } from '../types'
import { createLogger } from '../media/logger'

const log = createLogger('notify')

const flowTypeMap: Record<FlowType, [string, string]> = {
  supervision: ['#00d4ff', '监护制'],
  info_notice: ['#00ffcc', '信息通报'],
  // 中性兜底：flow_type 缺失或未登记时显示"流程"，绝不借用别的制度名
  unknown: ['#8899aa', '流程'],
}

// 制度类型归一化: 只接受已登记名, 其余(缺失/null/未登记)一律归到 unknown。
// 旧实现用 'supervision' 兜底, 任何缺 flow_type 的事件都会被误标成"监护制"
function normalizeFlowType(raw: unknown): FlowType {
  const key = typeof raw === 'string' ? raw : ''
  return key && key in flowTypeMap ? (key as FlowType) : 'unknown'
}

// 同一时刻内的显示序: 流程开始 → 子步骤判定 → 流程结束。
// 结束行排最后，避免"信息通报结束"盖在自己那批"未…"判定行前面（读起来像是先结束再判定）
const kindRank: Record<FlowKind, number> = { start: 0, step: 1, end: 2 }

// 同类流程序号: 一个视频里可能出现多个同类流程（如中间、结尾各一次"信息通报"），
// 逐行展示时必须能看出这行属于哪一个流程。按"类型内首次出现顺序"给每个 flow_id 编号，
// 键取 flowType#flowId（不同制度的 flow_id 各自计数，互不干扰）。
const flowSeqMap = new Map<string, number>()
const flowSeqCounter: Record<string, number> = {}

function flowSeqOf(flowType: FlowType, flowId: unknown): number {
  const id = flowId === undefined || flowId === null ? '' : String(flowId)
  if (!id) return 0
  const key = flowType + '#' + id
  const cached = flowSeqMap.get(key)
  if (cached !== undefined) return cached
  const seq = (flowSeqCounter[flowType] || 0) + 1
  flowSeqCounter[flowType] = seq
  flowSeqMap.set(key, seq)
  return seq
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
    // 静默收口（上一轮被中断 / 视频结束时流程仍活跃）: 后端给的结束时刻只是"最后观测
    // 时刻"而非真实结束，显示出来会和下一轮的"开始"挤在同一秒，故丢弃不显示
    if (!isStart && dt.silent === true) return
    // 结束时刻显式为 0/非法(后端 0 值收口)时丢弃该行：否则会被 localSec 兜底成
    // 很小的秒数，在片头冒出一行"xx结束"，看起来像"一开始就发生了流程"
    if (!isStart && dt.flow_end_sec !== undefined && Number(dt.flow_end_sec) <= 0) return
    const sec = Number(isStart
      ? (dt.flow_start_sec || src.localSec || src.sec || 0)
      : (dt.flow_end_sec || src.localSec || src.sec || 0))
    const flowType = normalizeFlowType(dt.flow_type ?? src.flowType)
    const [color, name] = flowTypeMap[flowType]
    // 设备码（九字码/8位码）：流程由"读出设备码"启动时，开始行直接带上码值，
    // 否则每行都是同样的"流程开始"，无法区分这一票对应哪台设备
    const deviceCode = typeof dt.device_code === 'string' ? dt.device_code.trim() : ''
    const text = name + (isStart ? '开始' : '结束') + (isStart && deviceCode ? ' · ' + deviceCode : '')
    const flowSeq = flowSeqOf(flowType, dt.flow_id ?? (src as { flow_id?: unknown }).flow_id)
    // 同一事件有两条推送链路（InferenceSync 自订阅 + 评估管理器转发），按"流程+文案+时刻"去重；
    // 带 flowSeq 比较是为了不把另一个同类流程的同文案行误并掉
    const existed = rawFlowEvents.value.find(
      e => !e.isStep && e.flowSeq === flowSeq && e.text === text && Math.abs(e.sec - sec) < 0.1)
    if (existed) return
    rawFlowEvents.value.push({
      sec: Number(sec.toFixed(2)), flowType, color, text,
      isStep: false, ok: null, kind: isStart ? 'start' : 'end', flowSeq,
    })
  }

  // 制度层实时下发的子步骤判定行（举手/请讲/关注/通报完毕/收到）
  function addFlowStep(d: unknown): void {
    const src = d as { localSec?: number; flowType?: FlowType; data?: Record<string, unknown> }
    const dt = src.data || {}
    const sec = Number(dt.step_sec || src.localSec || 0)
    const flowType = normalizeFlowType(dt.flow_type ?? src.flowType)
    const [color] = flowTypeMap[flowType]
    const text = String(dt.text || '')
    if (!text) return
    const ok = dt.ok === true ? true : dt.ok === false ? false : null
    const flowSeq = flowSeqOf(flowType, dt.flow_id ?? (src as { flow_id?: unknown }).flow_id)
    // 同一判定可能被两条推送链路各送一次（InferenceSync 自订阅 + 评估管理器转发）；
    // 去重键含 flowSeq：两个同类流程各自的"未举手"是两行，不能被并成一行
    const existed = rawFlowEvents.value.find(
      e => e.isStep && e.flowSeq === flowSeq && e.text === text && Math.abs(e.sec - sec) < 0.6)
    if (existed) return
    rawFlowEvents.value.push({
      sec: Number(sec.toFixed(2)), flowType, color, text,
      isStep: true, ok, kind: 'step', flowSeq,
    })
  }

  // 事件流: 严格同步, 与画面同一时刻(无提前量);
  // 同一时刻内按 开始 → 子步骤判定 → 结束 排列（不依赖推送到达顺序，两条链路乱序也稳）
  const flowEvents = computed(() =>
    rawFlowEvents.value
      .filter(e => e.sec <= _currentPlaybackSec.value)
      .slice()
      .sort((a, b) => (a.sec - b.sec) || (kindRank[a.kind] - kindRank[b.kind]))
  )

  // 刷新恢复: 回填缓存的流程事件行, 并把序号计数器推到已用最大值,
  // 避免恢复后若仍有新事件到达时与已恢复行重号
  function restore(items: FlowEvent[]): void {
    for (const item of items || []) {
      rawFlowEvents.value.push({ ...item })
      const used = flowSeqCounter[item.flowType] || 0
      if (item.flowSeq > used) flowSeqCounter[item.flowType] = item.flowSeq
    }
  }

  function reset(): void {
    people.value = { count: '--', alert: '就绪', alertColor: '#8899aa' }
    gaze.value = { hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 }
    rawFlowEvents.value = []
    // 流程序号随本轮清空: 新一轮从 ① 重新编号
    flowSeqMap.clear()
    for (const key of Object.keys(flowSeqCounter)) delete flowSeqCounter[key]
  }

  return { people, gaze, flowEvents, rawFlowEvents, addTracking, addGaze, addFlow, addFlowStep, restore, reset }
}
