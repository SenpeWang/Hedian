import { ref, reactive, computed } from 'vue'
import type { VoiceEntry, FlowEvent, SegCard, PeopleState, GazeState } from '../types'

export interface ProgressState {
  voice: number
  tracker: number
  gaze: number
  detail: string
}

export interface ParsedBatch {
  globalSec: number
  frames: Record<number, string>
  meta: Record<string, any>
}

const FRAME_MS = 1000 / 30

export function useWS() {
  const status = ref<'idle' | 'running' | 'done'>('idle')
  const statusText = ref('就绪')
  let socket: WebSocket | null = null
  let isFirstBatch = true

  // 单一原子响应式 State
  const unifiedBatch = reactive({
    globalSec: 0,
    frontUrl: '',
    popUrl: '',
    hasFront: false,
    hasPop: false,
    gaze: { hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 } as GazeState,
    people: { count: '--', alert: '就绪', alertColor: '#8899aa' } as PeopleState,
    progress: { voice: 0, tracker: 0, gaze: 0, detail: '' } as ProgressState,
  })

  const frameFront = computed(() => unifiedBatch.frontUrl)
  const framePop = computed(() => unifiedBatch.popUrl)
  const hasFrameFront = computed(() => unifiedBatch.hasFront)
  const hasFramePop = computed(() => unifiedBatch.hasPop)
  const gaze = unifiedBatch.gaze
  const people = unifiedBatch.people
  const progress = unifiedBatch.progress

  // 播放队列 + 高精度时间累加器出帧循环
  const playbackQueue: ParsedBatch[] = []
  let eofReceived = false
  let rafId: number | null = null
  let lastObjectUrls: Record<number, string> = {}
  let lastPlaybackReportTime = 0
  let lastAudioSyncTime = 0

  // 业务数据
  const voiceMap = reactive<Record<number, VoiceEntry>>({})
  const voiceEntries = computed(() =>
    Object.values(voiceMap).sort((a, b) => a.sec - b.sec)
  )
  const flowEvents = ref<FlowEvent[]>([])
  const segCards = ref<SegCard[]>([])
  const segScores = ref<number[]>([])
  const supN = ref(0)
  const ticketN = ref(0)
  const noticeN = ref(0)
  const completedFlows = new Set<string>()

  function fmt(sec: any) {
    if (sec === undefined || sec === null) return '00:00'
    const s = Math.floor(Number(sec))
    const m = Math.floor(s / 60)
    const r = s % 60
    return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0')
  }

  function parseBinaryPacket(buf: ArrayBuffer): ParsedBatch | null {
    try {
      if (buf.byteLength < 6) return null
      const view = new DataView(buf)
      let offset = 0
      view.getUint8(offset); offset += 1 // version
      const globalSec = view.getFloat32(offset, false); offset += 4
      const viewCount = view.getUint8(offset); offset += 1

      const frames: Record<number, string> = {}
      for (let i = 0; i < viewCount; i++) {
        if (offset + 5 > buf.byteLength) break
        const viewId = view.getUint8(offset); offset += 1
        const frameLen = view.getUint32(offset, false); offset += 4
        if (offset + frameLen > buf.byteLength) break
        const jpegBytes = new Uint8Array(buf, offset, frameLen)
        offset += frameLen
        const blob = new Blob([jpegBytes], { type: 'image/jpeg' })
        frames[viewId] = URL.createObjectURL(blob)
      }

      let meta: any = {}
      if (offset + 4 <= buf.byteLength) {
        const jsonLen = view.getUint32(offset, false); offset += 4
        if (offset + jsonLen <= buf.byteLength) {
          const jsonBytes = new Uint8Array(buf, offset, jsonLen)
          const jsonStr = new TextDecoder('utf-8').decode(jsonBytes)
          meta = JSON.parse(jsonStr)
        }
      }
      const finalGlobalSec = meta.globalSec !== undefined ? Number(meta.globalSec) : globalSec
      return { globalSec: finalGlobalSec, frames, meta }
    } catch (err) {
      console.error('二进制解析失败:', err)
      return null
    }
  }

  function connect() {
    if (socket) socket.close()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/data`)
    socket.binaryType = 'arraybuffer'

    socket.onopen = () => {
      status.value = 'running'
      statusText.value = '🟢 运行中'
      if (rafId === null) {
        startPlaybackLoop()
      }
    }
    socket.onmessage = (evt) => {
      try {
        if (evt.data instanceof ArrayBuffer) {
          const parsed = parseBinaryPacket(evt.data)
          if (!parsed) return
          playbackQueue.push(parsed)
        } else {
          const d = JSON.parse(evt.data)
          if (d.source === 'done') {
            if (d.tag === 'stop') {
              status.value = 'done'
              statusText.value = '🛑 已停止'
              stopPlaybackLoop()
            } else {
              eofReceived = true
            }
          } else if (d.tag === 'segment_report_stream') {
            handleDirectSegReportStream(d)
          } else if (d.tag === 'segment_report') {
            handleDirectSegReport(d)
          }
        }
      } catch (ex) { console.error('WS 处理异常:', ex) }
    }
    socket.onclose = () => {
      status.value = 'idle'
      statusText.value = '就绪'
      stopPlaybackLoop()
    }
    socket.onerror = () => {
      status.value = 'idle'
      statusText.value = '❌ 连接错误'
      stopPlaybackLoop()
    }
  }

  // 评估流式直推
  function handleDirectSegReportStream(d: any) {
    const dt = d.data || d
    const fid = String(dt.flow_id || dt.flowId || '')
    if (!fid) return
    const chunk = dt.chunk || ''
    let card = segCards.value.find(c => c.flowId === fid)
    if (!card) {
      card = {
        flowId: fid,
        flowType: dt.flow_type || 'supervision',
        score: 0,
        reportText: '',
        continueSec: dt.flow_continue_sec || '?',
        collapsed: false,
        streamBuffer: chunk,
        streaming: true,
      }
      segCards.value.push(card)
    } else {
      card.streamBuffer += chunk
      card.streaming = true
    }
  }

  // 评估完成直推
  function handleDirectSegReport(d: any) {
    const dt = d.data || d
    const fid = String(dt.flow_id || dt.flowId || '')
    if (!fid) return

    if (!completedFlows.has(fid)) {
      completedFlows.add(fid)
      segScores.value.push(dt.score || 0)
      const flowType = dt.flow_type || 'supervision'
      if (flowType === 'supervision') supN.value++
      else if (flowType === 'info_notice') noticeN.value++
      else ticketN.value++
    }

    let card = segCards.value.find(c => c.flowId === fid)
    if (!card) {
      card = {
        flowId: fid,
        flowType: dt.flow_type || 'supervision',
        score: dt.score || 0,
        reportText: dt.report_text || '',
        continueSec: dt.flow_continue_sec || '?',
        collapsed: false,
        streamBuffer: dt.report_text || '',
        streaming: false,
      }
      segCards.value.push(card)
    } else {
      card.score = dt.score || 0
      card.reportText = dt.report_text || ''
      card.streamBuffer = dt.report_text || card.streamBuffer
      card.streaming = false
    }
  }

  function startPipeline() {
    resetState()
    status.value = 'running'
    statusText.value = '🚀 启动中...'
    fetch('/start', { method: 'POST' }).then(() => {
      connect()
    }).catch(e => {
      console.error('启动失败:', e)
      status.value = 'idle'
    })
  }

  /**
   * 工业级高精度时间累加器出帧循环 (Time Accumulator Loop)
   * 
   * 根据 batch 的 globalSec 时间差精确出帧，完美兼容 60/120/144Hz 屏幕与不同视频源帧率，
   * 消除丢帧与顿挫，保持极致丝滑。
   */
  function startPlaybackLoop() {
    stopPlaybackLoop()
    let last = performance.now()
    let acc = 0
    let lastBatchSec = -1

    function loop(now: number) {
      acc += now - last
      last = now

      while (playbackQueue.length > 0) {
        const next = playbackQueue[0]
        let stepMs = FRAME_MS
        if (lastBatchSec >= 0 && next.globalSec >= lastBatchSec) {
          stepMs = Math.max(5, Math.min(500, (next.globalSec - lastBatchSec) * 1000))
        }
        if (acc < stepMs) break
        acc -= stepMs
        lastBatchSec = next.globalSec
        playOneBatch(playbackQueue.shift()!)
      }

      if (playbackQueue.length === 0) {
        acc = 0
        if (eofReceived) {
          status.value = 'done'
          statusText.value = '✅ 推理完成'
          rafId = null
          return
        }
      }
      rafId = requestAnimationFrame(loop)
    }
    rafId = requestAnimationFrame(loop)
  }

  function stopPlaybackLoop() {
    if (rafId !== null) {
      cancelAnimationFrame(rafId)
      rafId = null
    }
    const audioEl = document.getElementById('mainAudio') as HTMLAudioElement
    if (audioEl) {
      audioEl.pause()
      audioEl.removeAttribute('src')
      audioEl.load()
    }
    lastAudioSyncTime = 0
  }

  function playOneBatch(batch: ParsedBatch) {
    unifiedBatch.globalSec = batch.globalSec

    // 1. 首包触发音频播放
    if (isFirstBatch) {
      isFirstBatch = false
      const audioEl = document.getElementById('mainAudio') as HTMLAudioElement
      if (audioEl) {
        audioEl.src = '/api/audio/stream?t=' + Date.now()
        audioEl.currentTime = batch.globalSec || 0
        audioEl.playbackRate = 1.0
        audioEl.play().catch(e => console.warn('Audio play deferred:', e))
      }
    }

    // 2. 工业级三段式音频平滑伴音跟随
    syncAudioWithBatch(batch.globalSec)

    // 3. 上报播放进度给后端（驱动流程评价在达到播放时间点时触发）
    const nowSec = performance.now() / 1000.0
    if (socket && socket.readyState === WebSocket.OPEN && nowSec - lastPlaybackReportTime >= 0.5) {
      try {
        socket.send(JSON.stringify({
          type: 'playback_progress',
          current_sec: Number(batch.globalSec.toFixed(2)),
        }))
        lastPlaybackReportTime = nowSec
      } catch (err) {
        console.warn('上报播放进度失败:', err)
      }
    }

    // 4. 视频帧原子更新与 ObjectURL 内存管理
    const f = batch.frames
    if (f[0] !== undefined) {
      if (lastObjectUrls[0]) URL.revokeObjectURL(lastObjectUrls[0])
      lastObjectUrls[0] = f[0]
      unifiedBatch.frontUrl = f[0]
      unifiedBatch.hasFront = true
    }
    if (f[1] !== undefined) {
      if (lastObjectUrls[1]) URL.revokeObjectURL(lastObjectUrls[1])
      lastObjectUrls[1] = f[1]
      unifiedBatch.popUrl = f[1]
      unifiedBatch.hasPop = true
    }

    // 5. 元数据更新
    const d = batch.meta
    if (d.gaze?.[0]?.data) {
      const dt = d.gaze[0].data
      unifiedBatch.gaze.hasHeads = !!dt.has_heads
      unifiedBatch.gaze.headsCount = dt.heads_count || 0
      unifiedBatch.gaze.anyInRoi = !!dt.any_in_roi
      unifiedBatch.gaze.awayDuration = dt.away_duration || 0
    }
    if (d.voice && Array.isArray(d.voice)) {
      for (const v of d.voice) addVoice(v)
    }
    if (d.tracking && Array.isArray(d.tracking)) {
      for (const t of d.tracking) addTracking(t)
    }
    if (d.progress && Array.isArray(d.progress)) {
      for (const p of d.progress) updateProgress(p)
    }
    if (d.flow_start && Array.isArray(d.flow_start)) {
      for (const fe of d.flow_start) addFlow(fe, true)
    }
    if (d.flow_end && Array.isArray(d.flow_end)) {
      for (const fe of d.flow_end) addFlow(fe, false)
    }
  }

  function syncAudioWithBatch(globalSec: number) {
    const audioEl = document.getElementById('mainAudio') as HTMLAudioElement
    if (!audioEl || audioEl.paused || audioEl.ended) return

    const now = performance.now()
    if (now - lastAudioSyncTime < 200) return
    lastAudioSyncTime = now

    const diff = globalSec - audioEl.currentTime
    const absDiff = Math.abs(diff)

    if (absDiff > 1.5) {
      audioEl.currentTime = globalSec
      if (audioEl.playbackRate !== 1.0) audioEl.playbackRate = 1.0
      return
    }

    if (absDiff < 0.12) {
      if (audioEl.playbackRate !== 1.0) audioEl.playbackRate = 1.0
      return
    }

    if (diff > 0.12) {
      audioEl.playbackRate = 1.05
    } else {
      audioEl.playbackRate = 0.95
    }
  }

  function resetState() {
    isFirstBatch = true
    unifiedBatch.frontUrl = ''
    unifiedBatch.popUrl = ''
    unifiedBatch.hasFront = false
    unifiedBatch.hasPop = false
    unifiedBatch.globalSec = 0
    playbackQueue.length = 0
    eofReceived = false
    for (const k in lastObjectUrls) { URL.revokeObjectURL(lastObjectUrls[k]); delete lastObjectUrls[k] }
    stopPlaybackLoop()
    for (const k in voiceMap) delete voiceMap[Number(k)]
    flowEvents.value = []
    segCards.value = []
    segScores.value = []
    supN.value = 0; ticketN.value = 0; noticeN.value = 0
    completedFlows.clear()
    unifiedBatch.people.count = '--'; unifiedBatch.people.alert = '就绪'; unifiedBatch.people.alertColor = '#8899aa'
    unifiedBatch.gaze.hasHeads = false; unifiedBatch.gaze.headsCount = 0; unifiedBatch.gaze.anyInRoi = false; unifiedBatch.gaze.awayDuration = 0
    unifiedBatch.progress.voice = 0; unifiedBatch.progress.tracker = 0; unifiedBatch.progress.gaze = 0; unifiedBatch.progress.detail = ''
    lastAudioSyncTime = 0
  }

  function addVoice(d: any) {
    const dt = d.data || {}
    const sec = d.localSec || dt.sec || 0
    if (!voiceMap[sec]) {
      voiceMap[sec] = { sec, text: dt.text || '', keys: dt.keys || [] }
    } else if (dt.text) {
      voiceMap[sec].text = dt.text
      if (dt.keys) voiceMap[sec].keys = dt.keys
    }
  }

  function addTracking(d: any) {
    const dt = d.data || {}
    if (d.tag === 'PEOPLE_COUNT_UPDATE') {
      unifiedBatch.people.count = dt.count ?? '--'
      if (dt.state_alert) { unifiedBatch.people.alert = '⚠️ ' + dt.state_alert; unifiedBatch.people.alertColor = '#ff4d4d' }
      else { unifiedBatch.people.alert = '✅ 当前人数正常'; unifiedBatch.people.alertColor = '#00ff88' }
    }
  }

  const flowTypeMap: Record<string, [string, string]> = {
    supervision: ['#00d4ff', '监护制'],
    info_notice: ['#00ffcc', '信息通报'],
  }

  function addFlow(d: any, isStart: boolean) {
    const dt = d.data || d
    const sec = Number(isStart
      ? (dt.flow_start_sec || d.localSec || d.sec || 0)
      : (dt.flow_end_sec || d.localSec || d.sec || 0))
    const flowType = dt.flow_type || d.flowType || 'supervision'
    const [color, name] = flowTypeMap[flowType] || ['#ffaa00', '自唱票']
    
    const existing = flowEvents.value.find(e => e.flowType === flowType && Math.abs(e.sec - sec) < 0.1 && e.isStart === isStart)
    if (!existing) {
      flowEvents.value.push({
        sec: Number(sec.toFixed(2)),
        flowType,
        name, color, isStart,
      })
    }
  }

  function updateProgress(d: any) {
    const dt = d.data || {}
    const label = dt.label || d.label || ''
    const pct = Math.max(0, Math.min(100, dt.pct || 0))
    if (label === 'voice') unifiedBatch.progress.voice = pct
    else if (label === 'tracker') unifiedBatch.progress.tracker = pct
    else if (label === 'gaze') unifiedBatch.progress.gaze = pct
    else unifiedBatch.progress.detail = dt.detail || (dt.pct !== undefined ? pct.toFixed(1) + '%' : '')
  }

  function totalCount() { return segScores.value.reduce((a, b) => a + b, 0) }
  function avgScore() { return segScores.value.length > 0 ? (totalCount() / segScores.value.length).toFixed(1) : '-' }

  async function stopPipeline() {
    statusText.value = '⏳ 正在停止...'
    try {
      await fetch('/stop', { method: 'POST' })
    } catch (e) {
      console.error('停止请求失败:', e)
    }
    status.value = 'done'
    statusText.value = '🛑 已停止'
    stopPlaybackLoop()
  }

  return {
    status, statusText, startPipeline, stopPipeline, fmt,
    voiceEntries, people, gaze, flowEvents,
    segCards, segScores, supN, ticketN, noticeN,
    progress, totalCount, avgScore,
    frameFront, framePop,
    hasFrameFront, hasFramePop,
  }
}
