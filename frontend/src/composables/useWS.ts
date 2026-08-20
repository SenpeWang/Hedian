import { ref, reactive, computed, onBeforeUnmount } from 'vue'
import type {
  VoiceEntry, FlowEvent, SegCard, FlowType,
GazeState, PeopleState,
} from '../types'
import { TimeSeriesPool } from './useTimeSeriesPool'

// MSE 视频流缓冲: 串行 appendBuffer(等 updateend 再推下一段)
interface VisBuffer {
  push: (data: ArrayBuffer) => void
  end: () => void
}

function createVisBuffer(ms: MediaSource, codec: string, getClock: () => number): VisBuffer {
  let sb: SourceBuffer | null = null
  const queue: ArrayBuffer[] = []
  let pending = false
  let ended = false
  let endedWanted = false
  let trimming = false
  let failCount = 0

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
        // 队列排空后若已请求结束, 再 endOfStream(防末段丢失)
        if (endedWanted && !ended && queue.length === 0 && !pending && sb && !sb.updating) {
          ended = true
          try { if (ms.readyState === 'open') ms.endOfStream() } catch { /* 忽略 */ }
        }
      })
      sb.addEventListener('error', (e: any) => console.error('[mse] SourceBuffer error', e))
      pump()
    } catch (e) { console.error('[mse] addSourceBuffer 失败:', codec, e) }
  })

  // 清已播放旧数据防配额溢出: 按主时钟(非 video.currentTime)删前 8s, pop 失锁不乱
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
    const buf = queue[0]
    pending = true
    try {
      sb.appendBuffer(buf)
      queue.shift()          // 成功调用后才出队(失败留队重试, 不丢帧)
      failCount = 0
    } catch (e) {
      pending = false
      failCount++
      console.warn('appendBuffer 失败:', e)
      if (failCount > 3) { queue.shift(); failCount = 0; console.error('appendBuffer 重试超限, 丢弃该帧') }
    }
  }

  return {
    push: (data: ArrayBuffer) => { queue.push(data); pump() },
    end: () => {
      if (ended || endedWanted) return
      // 队列空且非 pending 才立即 endOfStream, 否则等 pump 排空后 updateend 触发
      if (queue.length === 0 && !pending && sb && !sb.updating) {
        ended = true
        try { if (ms.readyState === 'open') ms.endOfStream() } catch { /* 忽略 */ }
      } else {
        endedWanted = true
      }
    },
  }
}

export function useWS() {
  const status = ref<'idle' | 'starting' | 'running' | 'done' | 'stopped'>('idle')
  const isPlaying = computed(() => status.value === 'running' || status.value === 'starting')

  const currentPlaybackSec = ref(0) // 主时钟: 前端播放进度(front currentTime)
  const totalDuration = ref(0)
  const globalSec = ref(0) // 后端推理进度(进度条用, 超前播放)
  const viewSecs = ref({ front: 0, pop: 0, voice: 0 })
  let _vsLast = { front: 0, pop: 0, voice: 0 }, _vsClockLast = 0
  const playbackRate = ref(1.0)
  // 速率引擎: playbackRate=min(各视角增速,1.0), 慢路拖累(主等从); EMA 平滑防阶跃, 下限0.2
  // 停滞视角(d<=0)用 0 参与 min(真正拖累), 不跳过
  function _updateRate() {
    const now = performance.now()
    if (_vsClockLast === 0) { _vsClockLast = now; _vsLast = { ...viewSecs.value }; return }
    const dt = (now - _vsClockLast) / 1000
    if (dt < 2) return
    const vs = viewSecs.value
    const speeds: number[] = []
    for (const k of ['front', 'pop', 'voice'] as const) {
      const d = (vs[k] - _vsLast[k]) / dt
      speeds.push(Math.max(0, d))   // 停滞 0 参与 min, 不跳过(慢路真正拖累)
    }
    _vsClockLast = now; _vsLast = { ...vs }
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

  // ── WebSocket ──
  let socket: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0
  let lastReportSec = 0
  let lastReportReal = 0
  const pendingReports: number[] = []   // WS 非 OPEN 时暂存上报, onopen flush(防 wait_playback_reached 死锁)

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
    frontBuf = createVisBuffer(frontMS, 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"', () => currentPlaybackSec.value)

    popMS = new MediaSource()
    popMediaUrl.value = URL.createObjectURL(popMS)
    popBuf = createVisBuffer(popMS, 'video/mp4; codecs="avc1.42E01E"', () => currentPlaybackSec.value)
  }

  const _visCount = { front: 0, pop: 0 }
  function handleBinary(u8: Uint8Array) {
    if (u8.length < 2) return
    const channel = u8[0]      // 0=front, 1=pop
    const type = u8[1]         // 0=init, 1=media, 2=end
    if (channel !== 0 && channel !== 1) return   // 显式校验, 非法 channel 丢弃
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
      if (msg.status) { status.value = msg.status }
      return
    }
    if (msg.source === 'done' || (msg.meta && msg.meta.source === 'done')) {
      status.value = 'done'
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
      reconnectAttempts = 0
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
      // flush 重连期间暂存的上报
      while (pendingReports.length && socket) {
        const s = pendingReports.shift()!
        try { socket.send(JSON.stringify({ type: 'playback_progress', current_sec: Number(s.toFixed(2)) })) }
        catch { pendingReports.unshift(s); break }
      }
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
    reconnectAttempts++
    const delay = Math.min(30000, 1000 * 2 ** Math.min(reconnectAttempts, 5))   // 指数退避, 上限 30s
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connect() }, delay)
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

  // 设主时钟 + 上报后端(供评估 wait_playback_reached); WS 非 OPEN 时暂存待 flush
  function reportPlaybackProgress(currentSec: number) {
    currentPlaybackSec.value = currentSec
    const now = performance.now()
    if (Math.abs(currentSec - lastReportSec) < 0.5 && now - lastReportReal < 500) return
    lastReportSec = currentSec; lastReportReal = now
    const payload = JSON.stringify({ type: 'playback_progress', current_sec: Number(currentSec.toFixed(2)) })
    if (socket && socket.readyState === WebSocket.OPEN) {
      try { socket.send(payload) }
      catch { pendingReports.push(currentSec) }
    } else {
      pendingReports.push(currentSec)
    }
  }

  // computed 跟随主时钟 currentPlaybackSec: 状态量 getLatestAt(最新可得); 事件流 filter(已发生)
  const voiceEntries = computed(() =>
    Object.values(rawVoiceMap).filter(v => v.sec <= currentPlaybackSec.value + 3).sort((a, b) => a.sec - b.sec)
  )
  const flowEvents = computed(() =>
    rawFlowEvents.value.filter(e => e.sec <= currentPlaybackSec.value + 3)
  )
  const people = computed<PeopleState>(() => {
    const item = peoplePool.getLatestAt(currentPlaybackSec.value)
    return item ? item.state : { count: '--', alert: '就绪', alertColor: '#8899aa' }
  })
  const gaze = computed<GazeState>(() => {
    const item = gazePool.getLatestAt(currentPlaybackSec.value)
    return item ? item.state : { hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 }
  })
  // 进度条 = globalSec/totalDuration(推理进度, 超前播放); 仅 done 强制 100, stopped 保真实比值不跳 100
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
  const flowTypeMap: Record<FlowType, [string, string]> = { supervision: ['#00d4ff', '监护制'], self_ticket: ['#ffaa00', '自唱票'], info_notice: ['#00ffcc', '信息通报'] }
  function addFlow(d: any, isStart: boolean) {
    const dt = d.data || d
    const sec = Number(isStart ? (dt.flow_start_sec || d.localSec || d.sec || 0) : (dt.flow_end_sec || d.localSec || d.sec || 0))
    const flowType = (dt.flow_type || d.flowType || 'supervision') as FlowType
    const [color, name] = flowTypeMap[flowType]
    if (!rawFlowEvents.value.find(e => e.flowType === flowType && Math.abs(e.sec - sec) < 0.1 && e.isStart === isStart)) {
      rawFlowEvents.value.push({ sec: Number(sec.toFixed(2)), flowType, name, color, isStart })
    }
  }
  function handleReportEvent(msg: any) {
    const data = msg.data || {}
    const flowId = String(data.flow_id || msg.flow_id || '')
    const flowType = (data.flow_type || msg.flow_type || 'supervision') as FlowType
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

  // SegCard 展示态收口: collapsed 折叠由 useWS 持有修改权, 组件 emit 调用(不直改 prop)
  function toggleCard(flowId: string) {
    const card = segCards.value.find(c => c.flowId === flowId)
    if (card) card.collapsed = !card.collapsed
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
    _visCount.front = 0; _visCount.pop = 0
    lastReportSec = 0; lastReportReal = 0
    pendingReports.length = 0
    // 重建 MSE(新 objectURL,清空 buffer)
    frontMediaUrl.value = ''; popMediaUrl.value = ''
    initMSE()
  }

  function startPipeline() {
    resetState()
    status.value = 'starting'
    fetch('/start', { method: 'POST' }).then(() => {
      status.value = 'running'
    }).catch(err => {
      console.error('启动失败:', err)
      status.value = 'idle'
    })
  }

  async function stopPipeline() {
    try { await fetch('/stop', { method: 'POST' }) } catch (e) { console.error('停止请求失败:', e) }
    status.value = 'stopped'   // 用户停止不等于推理完成, 不跳 100%
  }

  // 组件卸载时清理资源(socket/MediaSource/blob URL), 防 HMR 累积泄漏
  onBeforeUnmount(() => {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
    if (socket) { socket.onclose = null; socket.onerror = null; socket.close(); socket = null }
    for (const ms of [frontMS, popMS]) { if (ms) { try { if (ms.readyState === 'open') ms.endOfStream() } catch { /* 忽略 */ } } }
    if (frontMediaUrl.value) { try { URL.revokeObjectURL(frontMediaUrl.value) } catch {} }
    if (popMediaUrl.value) { try { URL.revokeObjectURL(popMediaUrl.value) } catch {} }
  })

  function totalCount() { return segScores.value.reduce((a, b) => a + b, 0) }
  function avgScore() { return segScores.value.length > 0 ? (totalCount() / segScores.value.length).toFixed(1) : '-' }

  initMSE()
  connect()

  return {
    status, isPlaying, startPipeline, stopPipeline, resetState, fmt,
    voiceEntries, people, gaze, flowEvents,
    segCards, supN, ticketN, noticeN,
    progress, totalCount, avgScore, toggleCard,
    reportPlaybackProgress,
    frontMediaUrl, popMediaUrl,
    playbackRate, currentPlaybackSec,
  }
}
