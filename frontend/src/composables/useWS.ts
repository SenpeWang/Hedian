import { ref, reactive, computed } from 'vue'
import type {
  VoiceEntry, FlowEvent, SegCard,
GazeState, PeopleState,
} from '../types'

// ─────────────────────────────────────────────────────────────
// 时序数据池:按 localSec 升序维护,支持状态查询(不早于)
// 结构化面板数据(人数/凝视)按 video.currentTime 随机访问
// ─────────────────────────────────────────────────────────────
// ── 时序数据池:按 localSec 升序维护,支持两种查询模式 ──
//  1. 状态量状态栏(人数/凝视): getLatestAt(sec) 二分取 <=sec 最后一项, 持续显示最新可得状态
//  2. 事件流状态栏(语音/流程): 外部用 filter(<=sec) 取所有已发生事件, 累积显示历史
class TimeSeriesPool<T extends { localSec: number }> {
  private items: T[] = []

  clear() { this.items = [] }
  get length() { return this.items.length }

  insertSorted(item: T) {
    const s = item.localSec
    if (this.items.length === 0 || s >= this.items[this.items.length - 1].localSec) {
      this.items.push(item); return
    }
    let lo = 0, hi = this.items.length - 1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (this.items[mid].localSec < s) lo = mid + 1
      else hi = mid - 1
    }
    this.items.splice(lo, 0, item)
  }

  /** 状态量状态栏查询: 二分找 localSec<=sec 的最后一项(人数/凝视等持续显示的最新状态) */
  getLatestAt(sec: number): T | null {
    if (this.items.length === 0) return null
    let lo = 0, hi = this.items.length - 1, ans = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (this.items[mid].localSec <= sec) { ans = mid; lo = mid + 1 }
      else hi = mid - 1
    }
    return ans >= 0 ? this.items[ans] : this.items[0]
  }
}

// ─────────────────────────────────────────────────────────────
// MSE 视频流缓冲:串行 appendBuffer(等 updateend 再推下一段)
// ─────────────────────────────────────────────────────────────
interface VisBuffer {
  push: (data: ArrayBuffer) => void
  end: () => void
}

function createVisBuffer(ms: MediaSource, codec: string, getClock: () => number): VisBuffer {
  let sb: SourceBuffer | null = null
  const queue: ArrayBuffer[] = []
  let pending = false
  let ended = false
  let trimming = false

  ms.addEventListener('sourceopen', () => {
    try {
      sb = ms.addSourceBuffer(codec)
      console.log('[mse] addSourceBuffer OK', codec)
      sb.mode = 'sequence'
      sb.addEventListener('updateend', () => {
        pending = false
        if (trimming) { trimming = false; pump(); return }
        maybeTrim()
        pump()
      })
      sb.addEventListener('error', (e: any) => console.error('[mse] SourceBuffer error', e))
      pump()
    } catch (e) { console.error('[mse] addSourceBuffer 失败:', codec, e) }
  })

  // 清理已播放过的旧数据,防止 SourceBuffer 配额溢出(QuotaExceededError)
  // 保留当前播放点前 8s,移除更早的已播放区间
  function maybeTrim() {
    if (!sb || sb.updating) return
    const t = getClock()
    if (t < 10) return
    if (sb.buffered.length === 0) return
    const start = sb.buffered.start(0)
    const cut = t - 8
    if (start < cut) {
      trimming = true
      try { sb.remove(start, cut) } catch { trimming = false }
    }
  }

  function pump() {
    if (pending || !sb || queue.length === 0) return
    pending = true
    const buf = queue.shift()!
    try { sb.appendBuffer(buf) }
    catch (e) { pending = false; console.warn('appendBuffer 失败:', e) }
  }

  return {
    push: (data: ArrayBuffer) => { queue.push(data); pump() },
    end: () => {
      if (ended) return
      ended = true
      try { if (ms.readyState === 'open') ms.endOfStream() } catch { /* 忽略 */ }
    },
  }
}

export function useWS() {
  const status = ref<'idle' | 'starting' | 'running' | 'done'>('idle')
  const statusText = ref('就绪')
  const isPlaying = computed(() => status.value === 'running' || status.value === 'starting')

  // 主时钟:当前视频播放时间(秒),所有 UI 展示跟随它
  const currentPlaybackSec = ref(0)
  // 总时长(秒), 从 front video.duration 取, 用于算播放进度
  const totalDuration = ref(0)
  // globalSec: 后端全局推理进度(仅 gaze localSec 兜底用; 速率引擎已改用 viewSecs)
  const globalSec = ref(0)
  // 各视角整体进度(秒): front/pop/voice, 用于分别算视角实时速度取 min
  const viewSecs = ref({ front: 0, pop: 0, voice: 0 })
  let _vsLast = { front: 0, pop: 0, voice: 0 }, _vsClockLast = 0
  const playbackRate = ref(1.0)
  // 统一速率引擎: 各视角整体速度 v=Δsec/Δwall, playbackRate=min(v_front,v_pop,v_voice,1.0)
  // 视角整体速率(每帧串行链吞吐), 非模块拆分; min 是缓冲水位门控(主等从, 慢路拖累快路);
  // EMA 平滑避免主时钟阶跃(pop 追跳变目标超调), 下限0.2, 上限1.0绝不超
  function _updateRate() {
    const now = performance.now()
    if (_vsClockLast === 0) { _vsClockLast = now; _vsLast = { ...viewSecs.value }; return }
    const dt = (now - _vsClockLast) / 1000
    if (dt < 2) return  // 至少2s算一次, 避免抖动
    const vs = viewSecs.value
    const speeds: number[] = []
    for (const k of ['front', 'pop', 'voice'] as const) {
      const d = (vs[k] - _vsLast[k]) / dt  // 视角整体 sec/sec 增速 = 实时倍率
      if (d > 0) speeds.push(d)
    }
    _vsClockLast = now; _vsLast = { ...vs }
    if (speeds.length === 0) return
    // 最慢视角决定整体: min(各视角速度, 1.0), 下限0.2; EMA 平滑防阶跃
    const target = Math.min(1.0, Math.max(0.2, Math.min(...speeds)))
    playbackRate.value = playbackRate.value * 0.6 + target * 0.4
  }

  // ── 时序数据池(结构化面板:人数/凝视)──
  const peoplePool = new TimeSeriesPool<{ localSec: number; state: PeopleState }>()
  const gazePool = new TimeSeriesPool<{ localSec: number; state: GazeState }>()

  // ── 评估/事件数据(ws 推送,高实时)──
  const rawVoiceMap = reactive<Record<number, VoiceEntry>>({})
  const rawFlowEvents = ref<FlowEvent[]>([])
  const segCards = ref<SegCard[]>([])
  const segScores = ref<number[]>([])
  const supN = ref(0)
  const ticketN = ref(0)
  const noticeN = ref(0)
  const completedFlows = new Set<string>()

  // ── MSE 视频流(front 带音频 / pop 静音)──
  const frontMediaUrl = ref('')
  const popMediaUrl = ref('')
  let frontMS: MediaSource | null = null
  let popMS: MediaSource | null = null
  let frontBuf: VisBuffer | null = null
  let popBuf: VisBuffer | null = null
  // 各视角 video.currentTime 取值函数(由 App 注入,用于 SourceBuffer trim 清理已播放数据)
  let frontClockFn: () => number = () => 0
  let popClockFn: () => number = () => 0
  // totalDuration 现由后端 batch 带(self.duration, 稳定), 不再依赖前端 video.duration
  function setClockFns(front: () => number, pop: () => number) {
    frontClockFn = front; popClockFn = pop
  }

  // ── WebSocket ──
  let socket: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let lastReportSec = 0
  let lastReportReal = 0

  function getWsUrl(): string {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${location.host}/ws/data`
  }

  function initMSE() {
    console.log('[mse] initMSE start')
    for (const ms of [frontMS, popMS]) { if (ms) { try { if (ms.readyState === 'open') ms.endOfStream() } catch {} } }
    if (frontMediaUrl.value) { try { URL.revokeObjectURL(frontMediaUrl.value) } catch {} }
    if (popMediaUrl.value) { try { URL.revokeObjectURL(popMediaUrl.value) } catch {} }
    // front:含音频轨;pop:仅视频
    frontMS = new MediaSource()
    frontMediaUrl.value = URL.createObjectURL(frontMS)
    frontBuf = createVisBuffer(frontMS, 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"', frontClockFn)

    popMS = new MediaSource()
    popMediaUrl.value = URL.createObjectURL(popMS)
    popBuf = createVisBuffer(popMS, 'video/mp4; codecs="avc1.42E01E"', popClockFn)
  }

  const _visCount = { front: 0, pop: 0 }
  function handleBinary(u8: Uint8Array) {
    if (u8.length < 2) return
    const channel = u8[0]      // 0=front, 1=pop
    const type = u8[1]         // 0=init, 1=media, 2=end
    const data = u8.subarray(2).slice().buffer
    const view = channel === 0 ? 'front' : 'pop'
    const buf = channel === 0 ? frontBuf : popBuf
    if (!buf) { console.warn('[vis] no buf for', view); return }
    if (type === 0) console.log('[vis] INIT', view, 'len=', u8.length)
    if (type === 2) { console.log('[vis] END', view, 'count=', _visCount[view]); buf.end(); return }
    _visCount[view]++
    if (_visCount[view] === 1 || _visCount[view] % 50 === 0) console.log('[vis] media', view, '#', _visCount[view], 'len=', u8.length)
    buf.push(data)
  }

  function handleJson(msg: any) {
    if (!msg) return
    // 连接时后端补发的状态快照: 拿到稳定 totalDuration(不依赖 video 加载)+当前 globalSec
    if (msg.source === 'status') {
      if (msg.totalDuration && msg.totalDuration > 0) totalDuration.value = msg.totalDuration
      if (msg.globalSec != null && msg.globalSec > 0) globalSec.value = msg.globalSec
      if (msg.status) { status.value = msg.status; if (msg.status === 'done') statusText.value = '✅ 推理完成' }
      return
    }
    if (msg.source === 'done' || (msg.meta && msg.meta.source === 'done')) {
      status.value = 'done'
      statusText.value = '✅ 推理完成'
      // 推理完成: 进度满(不依赖最后 batch 的 globalSec 是否到 total)
      if (totalDuration.value > 0) globalSec.value = totalDuration.value
      return
    }
    if (msg.source === 'segment_report_stream' || msg.source === 'segment_report' || msg.type === 'report_stream' || msg.type === 'report' || msg.source === 'stream' || msg.source === 'report') {
      handleReportEvent(msg); return
    }
    handleBatchEvent(msg)
  }

  function connect() {
    if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) return
    try { socket = new WebSocket(getWsUrl()) } catch { scheduleReconnect(); return }
    socket.binaryType = 'arraybuffer'

    socket.onopen = () => {
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
      if (status.value === 'idle') statusText.value = '就绪 (已连接)'
    }
    socket.onmessage = (evt) => {
      try {
        if (evt.data instanceof ArrayBuffer) {
          handleBinary(new Uint8Array(evt.data))
        } else if (typeof evt.data === 'string') {
          handleJson(JSON.parse(evt.data))
        }
      } catch (err) { console.warn('处理 WebSocket 消息失败:', err) }
    }
    socket.onerror = () => {}
    socket.onclose = () => { socket = null; scheduleReconnect() }
  }

  function scheduleReconnect() {
    if (reconnectTimer) return
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connect() }, 2000)
  }

  // ws batch 中的低频事件(评估/flow/progress/语音/人数/凝视)
  function handleBatchEvent(raw: any) {
    if (raw.globalSec != null) globalSec.value = raw.globalSec
    if (raw.totalDuration && raw.totalDuration > 0) totalDuration.value = raw.totalDuration
    if (raw.sourceTimes) { const st = raw.sourceTimes; viewSecs.value = { front: st.front ?? viewSecs.value.front, pop: st.pop ?? viewSecs.value.pop, voice: st.voice ?? viewSecs.value.voice }; _updateRate() }
    const d = raw.meta || raw
    if (d.flow_start && Array.isArray(d.flow_start)) for (const fe of d.flow_start) addFlow(fe, true)
    if (d.flow_end && Array.isArray(d.flow_end)) for (const fe of d.flow_end) addFlow(fe, false)
    if (d.voice && Array.isArray(d.voice)) for (const v of d.voice) addVoice(v)
    if (d.tracking && Array.isArray(d.tracking)) for (const t of d.tracking) addTracking(t)
    if (d.gaze && Array.isArray(d.gaze) && d.gaze[0]?.data) {
      const dt = d.gaze[0].data
      gazePool.insertSorted({
        localSec: d.gaze[0].localSec ?? raw.globalSec ?? 0,
        state: { hasHeads: !!dt.has_heads, headsCount: dt.heads_count || 0, anyInRoi: !!dt.any_in_roi, awayDuration: dt.away_duration || 0 }
      })
    }
  }

  // ── 播放进度上报(驱动主时钟 + 通知后端评估时机)──
  function reportPlaybackProgress(currentSec: number) {
    currentPlaybackSec.value = currentSec
    const now = performance.now()
    if (Math.abs(currentSec - lastReportSec) < 0.5 && now - lastReportReal < 500) return
    lastReportSec = currentSec; lastReportReal = now
    if (socket && socket.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ type: 'playback_progress', current_sec: Number(currentSec.toFixed(2)) })) }
      catch (err) { console.warn('上报播放进度失败:', err) }
    }
  }

  // ── computed: 所有 UI 严格跟随 currentPlaybackSec(前端播放时刻) ──
  //  分两类: 状态量状态栏(getLatestAt 最新值) / 事件流状态栏(filter 已发生全部)

  // [事件流状态栏] 语音转录: filter 累积所有已发生转录
  const voiceEntries = computed(() =>
    Object.values(rawVoiceMap).filter(v => v.sec <= currentPlaybackSec.value + 3).sort((a, b) => a.sec - b.sec)
  )
  // [事件流状态栏] 流程事件: filter 累积所有已发生流程开始/结束
  const flowEvents = computed(() =>
    rawFlowEvents.value.filter(e => e.sec <= currentPlaybackSec.value + 3)
  )
  // [状态量状态栏] 监控室人数: getLatestAt 取当前时刻最新可得人数, 持续显示
  const people = computed<PeopleState>(() => {
    const item = peoplePool.getLatestAt(currentPlaybackSec.value)
    return item ? item.state : { count: '--', alert: '就绪', alertColor: '#8899aa' }
  })
  // [状态量状态栏] 凝视状态: getLatestAt 取当前时刻最新可得凝视, 持续显示
  const gaze = computed<GazeState>(() => {
    const item = gazePool.getLatestAt(currentPlaybackSec.value)
    return item ? item.state : { hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 }
  })
  // 推理进度: 后端 globalSec(各视角min进度)/总时长, 不依赖前端播放, 持续更新; 完成时(done)直接 100%
  const progress = computed(() => status.value === 'done' ? 100 : (totalDuration.value > 0 ? Math.min(100, globalSec.value / totalDuration.value * 100) : 0))

  function fmt(s?: number): string {
    if (s === undefined || s === null || isNaN(s)) return '00:00'
    const m = Math.floor(s / 60), sec = Math.floor(s % 60)
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }

  // ── 事件解析 ──
  function addVoice(d: any) {
    const dt = d.data || {}
    const sec = d.localSec || dt.sec || 0
    if (!rawVoiceMap[sec]) rawVoiceMap[sec] = { sec, text: dt.text || '', keys: dt.keys || [] }
    else if (dt.text) { rawVoiceMap[sec].text = dt.text; if (dt.keys) rawVoiceMap[sec].keys = dt.keys }
  }
  function addTracking(d: any) {
    if (d.tag !== 'PEOPLE_COUNT_UPDATE') return
    const dt = d.data || {}
    peoplePool.insertSorted({
      localSec: d.localSec || 0,
      state: { count: dt.count ?? '--', alert: dt.state_alert ? '⚠️ ' + dt.state_alert : '✅ 当前人数正常', alertColor: dt.state_alert ? '#ff4d4d' : '#00ff88' }
    })
  }
  const flowTypeMap: Record<string, [string, string]> = { supervision: ['#00d4ff', '监护制'], info_notice: ['#00ffcc', '信息通报'] }
  function addFlow(d: any, isStart: boolean) {
    const dt = d.data || d
    const sec = Number(isStart ? (dt.flow_start_sec || d.localSec || d.sec || 0) : (dt.flow_end_sec || d.localSec || d.sec || 0))
    const flowType = dt.flow_type || d.flowType || 'supervision'
    const [color, name] = flowTypeMap[flowType] || ['#ffaa00', '自唱票']
    if (!rawFlowEvents.value.find(e => e.flowType === flowType && Math.abs(e.sec - sec) < 0.1 && e.isStart === isStart)) {
      rawFlowEvents.value.push({ sec: Number(sec.toFixed(2)), flowType, name, color, isStart })
    }
  }
  function handleReportEvent(msg: any) {
    const data = msg.data || {}
    const flowId = String(data.flow_id || msg.flow_id || '')
    const flowType = data.flow_type || msg.flow_type || 'supervision'
    const reportText = data.report_text || data.text || msg.text || ''
    const score = Number(data.score !== undefined ? data.score : 0)
    const continueSec = data.continue_sec || data.duration || 0
    if (!flowId) return
    let card = segCards.value.find(c => c.flowId === flowId)
    // token 到达驱动: segment_report_stream 的 data.chunk 按到达累积进 streamBuffer
    const chunk = data.chunk || ''
    if (!card) {
      card = { flowId, flowType, score, reportText: reportText || chunk, continueSec, collapsed: false, streamBuffer: chunk || reportText, streaming: msg.source === 'segment_report_stream' || msg.type === 'report_stream' || msg.source === 'stream' }
      segCards.value.push(card)
    } else {
      if (score > 0) card.score = score
      if (continueSec) card.continueSec = continueSec
      // 流式 chunk 累积(按 token 到达逐字); 终态用完整 report_text 覆盖
      if (chunk && (msg.source === 'segment_report_stream' || msg.type === 'report_stream')) {
        card.streamBuffer += chunk
      }
      if (reportText && (msg.source === 'segment_report' || msg.type === 'report' || msg.source === 'report')) {
        card.reportText = reportText
        // 不覆盖 streamBuffer、不设 streaming=false: 让 typewriter 自然逐字到 streamBuffer 末尾再停
        // (segment_report 到达时若覆盖会一次性全显示, 中断流式)
      }
    }
    if ((msg.source === 'segment_report' || msg.type === 'report' || msg.source === 'report') && !completedFlows.has(flowId)) {
      completedFlows.add(flowId)
      if (score > 0) segScores.value.push(score)
      if (flowType === 'supervision') supN.value++
      else if (flowType === 'self_ticket') ticketN.value++
      else if (flowType === 'info_notice') noticeN.value++
    }
  }

  // ── 生命周期 ──
  function resetState() {
    peoplePool.clear(); gazePool.clear()
    currentPlaybackSec.value = 0
    totalDuration.value = 0
    globalSec.value = 0; playbackRate.value = 1.0; viewSecs.value = { front: 0, pop: 0, voice: 0 }; _vsLast = { front: 0, pop: 0, voice: 0 }; _vsClockLast = 0
    for (const k in rawVoiceMap) delete rawVoiceMap[Number(k)]
    rawFlowEvents.value = []; segCards.value = []; segScores.value = []
    supN.value = 0; ticketN.value = 0; noticeN.value = 0; completedFlows.clear()
    // 重建 MSE(新 objectURL,清空 buffer)
    frontMediaUrl.value = ''; popMediaUrl.value = ''
    initMSE()
  }

  function startPipeline() {
    resetState()
    status.value = 'starting'
    statusText.value = '⚡ 正在启动...'
    fetch('/start', { method: 'POST' }).then(() => {
      status.value = 'running'
      statusText.value = '▶ 推理运行中'
    }).catch(err => {
      console.error('启动失败:', err)
      status.value = 'idle'
      statusText.value = '❌ 启动失败'
    })
  }

  async function stopPipeline() {
    statusText.value = '⏳ 正在停止...'
    try { await fetch('/stop', { method: 'POST' }) } catch (e) { console.error('停止请求失败:', e) }
    status.value = 'done'
    statusText.value = '🛑 已停止'
  }

  function totalCount() { return segScores.value.reduce((a, b) => a + b, 0) }
  function avgScore() { return segScores.value.length > 0 ? (totalCount() / segScores.value.length).toFixed(1) : '-' }

  initMSE()
  connect()

  return {
    status, statusText, isPlaying, startPipeline, stopPipeline, resetState, fmt,
    voiceEntries, people, gaze, flowEvents,
    segCards, segScores, supN, ticketN, noticeN,
    progress, totalCount, avgScore,
    reportPlaybackProgress,
    frontMediaUrl, popMediaUrl,
    setClockFns, initMSE,
    playbackRate, globalSec, viewSecs, totalDuration,
  }
}
