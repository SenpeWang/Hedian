import { ref, reactive, computed } from 'vue'

export interface ProgressState {
  voice: number
  tracker: number
  gaze: number
  detail: string
}

// 播放队列元素：一个对齐好的 batch（含三路视频帧 URL + JSON 元数据）
interface ParsedBatch {
  globalSec: number
  frames: Record<number, string>  // view_id (0=front/1=bup/2=pop) -> ObjectURL
  meta: any
}

// 30fps 固定步长出帧（兼容 60/120Hz 显示器：rAF + 累加器）
const FRAME_MS = 1000 / 30

export function useWS() {
  const status = ref<'idle' | 'starting' | 'running' | 'done'>('idle')
  const statusText = ref('就绪')
  let socket: WebSocket | null = null
  let isFirstBatch = true

  // ── 零冗余：单一原子响应式 State ──
  const unifiedBatch = reactive({
    globalSec: 0,
    frontUrl: '',
    bupUrl: '',
    popUrl: '',
    hasFront: false,
    hasBup: false,
    hasPop: false,
    gaze: { hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 },
    people: { count: '--', alert: '就绪', alertColor: '#8899aa' },
    progress: { voice: 0, tracker: 0, gaze: 0, detail: '' } as ProgressState,
  })

  // 兼容 template 解构别名
  const frameFront = computed(() => unifiedBatch.frontUrl)
  const frameBup = computed(() => unifiedBatch.bupUrl)
  const framePop = computed(() => unifiedBatch.popUrl)
  const hasFrameFront = computed(() => unifiedBatch.hasFront)
  const hasFrameBup = computed(() => unifiedBatch.hasBup)
  const hasFramePop = computed(() => unifiedBatch.hasPop)
  const gaze = unifiedBatch.gaze
  const people = unifiedBatch.people
  const progress = unifiedBatch.progress

  // ── 播放队列 + rAF 30fps 出帧 ──
  const playbackQueue: ParsedBatch[] = []
  let eofReceived = false
  let rafId: number | null = null
  // ObjectURL 内存管理：上一帧各视角 URL，挂载新帧前撤销旧帧
  let lastObjectUrls: Record<number, string> = {}

  // ── 数据列表 ──
  const voiceMap = reactive<Record<number, any>>({})
  const voiceEntries = computed(() =>
    Object.values(voiceMap).sort((a: any, b: any) => a.sec - b.sec)
  )
  const flowEvents = ref<any[]>([])
  const segCards = ref<any[]>([])
  const segScores = ref<number[]>([])
  const supN = ref(0), ticketN = ref(0), noticeN = ref(0)
  const completedFlows = new Set<string>()

  function fmt(sec: any) {
    if (sec === undefined || sec === null) return '00:00'
    const s = Math.floor(Number(sec))
    const m = Math.floor(s / 60)
    const r = s % 60
    return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0')
  }

  function connect(onConnected?: () => void) {
    if (socket) socket.close()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/data`)
    socket.binaryType = 'arraybuffer'

    socket.onopen = () => {
      onConnected?.()
      // 启动 30fps 播放循环
      if (rafId === null) {
        startPlaybackLoop()
      }
    }
    socket.onmessage = (evt) => {
      try {
        if (evt.data instanceof ArrayBuffer) {
          // 收到统一二进制消息：解包
          const parsed = parseBinaryPacket(evt.data)
          if (!parsed) return
          const src = parsed.meta?.source
          // 直推单事件（segment_report）：立即处理，不进播放队列
          // 判定依据：meta 顶层有 source 字段而无 globalSec（非对齐 batch）

          // 正常对齐 batch：入队按 30fps 出帧
          // 已渲染的 batch 由 rAF 循环 shift() 出队清除，ObjectURL 在下一帧覆盖时 revoke
          playbackQueue.push(parsed)
        } else {
          // 收到文本消息：done 哨兵或直推数据(如 segment_report_stream)
          const d = JSON.parse(evt.data)
          if (d.source === 'done') {
            if (d.tag === 'stop') {
              status.value = 'done'
              statusText.value = '🛑 已停止'
              stopPlaybackLoop()
            } else {
              eofReceived = true  // 不立即置 done，等队列排空
            }
          } else if (d.tag === 'segment_report_stream') {
            bufferSegReportStream(d)
            checkUnlockReports(unifiedBatch.globalSec)
          } else if (d.tag === 'segment_report') {
            bufferSegReport(d)
            checkUnlockReports(unifiedBatch.globalSec)
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
      statusText.value = '就绪'
    }
  }

  // ── 解析统一二进制消息协议 ──
  // [1B version][4B globalSec float32][1B view_count]
  // 重复 view_count 次: [1B view_id][4B frame_len][JPEG bytes]
  // [4B json_len][JSON bytes]
  function parseBinaryPacket(buf: ArrayBuffer): ParsedBatch | null {
    if (buf.byteLength < 6) return null
    const view = new DataView(buf)
    let offset = 0
    // version (忽略，当前仅 v1)
    view.getUint8(offset); offset += 1
    // globalSec
    const globalSec = view.getFloat32(offset); offset += 4
    // view_count
    const viewCount = view.getUint8(offset); offset += 1

    const frames: ParsedBatch['frames'] = {}
    for (let i = 0; i < viewCount; i++) {
      if (offset + 5 > buf.byteLength) break
      const viewId = view.getUint8(offset); offset += 1
      const frameLen = view.getUint32(offset); offset += 4
      if (offset + frameLen > buf.byteLength) break
      // 提取 raw JPEG 字节并创建 Blob URL（零 Base64 解码）
      const jpegBlob = new Blob([buf.slice(offset, offset + frameLen)], { type: 'image/jpeg' })
      frames[viewId] = URL.createObjectURL(jpegBlob)
      offset += frameLen
    }

    // 尾部 JSON 元数据
    let meta: any = {}
    if (offset + 4 <= buf.byteLength) {
      const jsonLen = view.getUint32(offset); offset += 4
      if (offset + jsonLen <= buf.byteLength) {
        const jsonStr = new TextDecoder().decode(new Uint8Array(buf, offset, jsonLen))
        meta = JSON.parse(jsonStr)
      }
    }
    // meta 若含 globalSec 优先用 meta 的（更精确，float32 可能丢精度）
    const finalGlobalSec = meta.globalSec !== undefined ? Number(meta.globalSec) : globalSec
    return { globalSec: finalGlobalSec, frames, meta }
  }

  // ── 30fps 固定步长播放循环 ──
  function startPlaybackLoop() {
    let last = performance.now()
    let acc = 0
    function loop(now: number) {
      acc += now - last
      last = now
      // 消化累积时间，每 FRAME_MS 出一帧
      while (acc >= FRAME_MS && playbackQueue.length > 0) {
        playOneBatch(playbackQueue.shift()!)
        acc -= FRAME_MS
      }
      // underrun：队列空时不累积，避免追上后突发倍速
      if (playbackQueue.length === 0) {
        acc = 0
        // EOF + 队列排空 → 真正完成
        if (eofReceived) {
          status.value = 'done'
          statusText.value = '✅ 完成'
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
    lastRealTime = 0
    lastGlobalSec = 0
  }

  // ── 出帧：应用一个 batch 到响应式状态 ──
  function playOneBatch(batch: ParsedBatch) {
    unifiedBatch.globalSec = batch.globalSec

    // 1. 首包触发音频播放
    if (isFirstBatch) {
      isFirstBatch = false
      const audioEl = document.getElementById('mainAudio') as HTMLAudioElement
      if (audioEl) {
        audioEl.src = '/api/audio/stream?t=' + Date.now()
        audioEl.currentTime = batch.globalSec || 0
        audioEl.play().catch(e => console.warn('Audio play deferred:', e))
      }
    }

    // 2. 5 秒滑动窗口倍速软调节
    syncAudioWithBatch(batch.globalSec)

    // 3. 挂载视频帧（撤销上一帧 ObjectURL 后挂新帧）
    const f = batch.frames
    if (f[0] !== undefined) {
      if (lastObjectUrls[0]) URL.revokeObjectURL(lastObjectUrls[0])
      lastObjectUrls[0] = f[0]
      unifiedBatch.frontUrl = f[0]; unifiedBatch.hasFront = true
    }
    if (f[1] !== undefined) {
      if (lastObjectUrls[1]) URL.revokeObjectURL(lastObjectUrls[1])
      lastObjectUrls[1] = f[1]
      unifiedBatch.bupUrl = f[1]; unifiedBatch.hasBup = true
    }
    if (f[2] !== undefined) {
      if (lastObjectUrls[2]) URL.revokeObjectURL(lastObjectUrls[2])
      lastObjectUrls[2] = f[2]
      unifiedBatch.popUrl = f[2]; unifiedBatch.hasPop = true
    }

    // 4. 应用 JSON 元数据（gaze/voice/tracking/progress/flow）
    const d = batch.meta
    if (d.gaze?.[0]?.data) {
      const dt = d.gaze[0].data
      gaze.hasHeads = !!dt.has_heads
      gaze.headsCount = dt.heads_count || 0
      gaze.anyInRoi = !!dt.any_in_roi
      gaze.awayDuration = dt.away_duration || 0
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
    // 8. 流式打字机渲染 Qwen 评估报告 chunk 碎片
    if (d.segment_report_stream && Array.isArray(d.segment_report_stream)) {
      for (const s of d.segment_report_stream) bufferSegReportStream(s)
    }
    // 9. 最终评估报告（包含分数与完整文本）
    if (d.segment_report && Array.isArray(d.segment_report)) {
      for (const r of d.segment_report) bufferSegReport(r)
    }

    // 10. 画面渲染滴答同步解锁评估卡片（视效画面物理到达流程结束点时打字吐流）
    checkUnlockReports(batch.globalSec)
  }

  let lastGlobalSec = 0
  let lastRealTime = 0
  let smoothedRate = 1.0

  function syncAudioWithBatch(globalSec: number) {
    const audioEl = document.getElementById('mainAudio') as HTMLAudioElement
    if (!audioEl) return

    const now = performance.now() / 1000.0
    if (lastRealTime > 0) {
      const dSec = globalSec - lastGlobalSec
      const dTime = now - lastRealTime

      if (dTime > 0.01 && dSec >= 0) {
        // 实时测算前端每一个解码 Batch 渲染的物理速率 (dSec / dTime)
        const rawRate = dSec / dTime
        if (dSec === 0) {
          audioEl.pause()  // 可视化卡顿/等待，声音实时暂停！
        } else {
          if (audioEl.paused) audioEl.play().catch(e => console.warn('Audio play error:', e))
          // EMA 动态锁步，实现声音播放自适应快慢使用当前前端解码可视化的速度
          smoothedRate = smoothedRate * 0.6 + rawRate * 0.4
          audioEl.playbackRate = Math.max(0.1, Math.min(2.5, smoothedRate))

          // 强误差硬校准 (当音频与 Batch 时间差超 0.8s 时微调指针)
          const timeDiff = globalSec - audioEl.currentTime
          if (Math.abs(timeDiff) > 0.8) {
            audioEl.currentTime = globalSec
          }
        }
      }
    }
    lastGlobalSec = globalSec
    lastRealTime = now
  }

  function startPipeline() {
    status.value = 'starting'
    statusText.value = '🚀 启动中…'
    resetState()

    connect(() => {
      status.value = 'running'
      fetch('/start', { method: 'POST' })
        .then(r => r.json())
        .catch(err => {
          console.error('启动失败:', err)
          statusText.value = '❌ 启动失败'
          status.value = 'idle'
        })
    })
  }

  function resetState() {
    isFirstBatch = true
    unifiedBatch.frontUrl = ''; unifiedBatch.bupUrl = ''; unifiedBatch.popUrl = ''
    unifiedBatch.hasFront = false; unifiedBatch.hasBup = false; unifiedBatch.hasPop = false
    // 清队列 + 撤销所有 ObjectURL + 停播放循环
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
    people.count = '--'; people.alert = '就绪'; people.alertColor = '#8899aa'
    gaze.hasHeads = false; gaze.headsCount = 0; gaze.anyInRoi = false; gaze.awayDuration = 0
    progress.voice = 0; progress.tracker = 0; progress.gaze = 0; progress.detail = ''
    lastGlobalSec = 0; lastRealTime = 0; smoothedRate = 1.0
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

    const keys = Object.keys(voiceMap).map(Number).sort((a, b) => a - b)
    if (keys.length > 200) {
      const toRemove = keys.length - 200
      for (let i = 0; i < toRemove; i++) {
        delete voiceMap[keys[i]]
      }
    }
  }

  function addTracking(d: any) {
    const dt = d.data || {}
    if (d.tag === 'PEOPLE_COUNT_UPDATE') {
      people.count = dt.count ?? '--'
      if (dt.state_alert) { people.alert = '⚠️ ' + dt.state_alert; people.alertColor = '#ff4d4d' }
      else { people.alert = '✅ 当前人数正常'; people.alertColor = '#00ff88' }
    }
  }

  const flowTypeMap: Record<string, [string, string]> = {
    supervision: ['#00d4ff', '监护制'],
    info_notice: ['#00ffcc', '信息通报'],
  }

  // ── 评价报告缓冲区：解耦后端直推，绑定画面出帧进度解锁 ──
  interface BufferedReport {
    flowId: string
    flowType: string
    endSec: number
    continueSec: string
    score: number
    reportText: string
    streamBuffer: string
    unlocked: boolean
  }
  const flowReportBuffer: Record<string, BufferedReport> = {}

  function bufferSegReportStream(d: any) {
    const dt = d.data || d
    const fid = String(dt.flow_id || dt.flowId || '')
    if (!fid) return
    if (!flowReportBuffer[fid]) {
      flowReportBuffer[fid] = {
        flowId: fid,
        flowType: dt.flow_type || 'supervision',
        endSec: dt.flow_end_sec || dt.end_sec || 0,
        continueSec: dt.flow_continue_sec || '?',
        score: 0,
        reportText: '',
        streamBuffer: '',
        unlocked: false,
      }
    }
    const chunk = dt.chunk || ''
    flowReportBuffer[fid].streamBuffer += chunk
  }

  function bufferSegReport(d: any) {
    const dt = d.data || d
    const fid = String(dt.flow_id || dt.flowId || '')
    if (!fid) return
    if (!flowReportBuffer[fid]) {
      flowReportBuffer[fid] = {
        flowId: fid,
        flowType: dt.flow_type || 'supervision',
        endSec: dt.flow_end_sec || dt.end_sec || 0,
        continueSec: dt.flow_continue_sec || '?',
        score: dt.score || 0,
        reportText: dt.report_text || '',
        streamBuffer: dt.report_text || '',
        unlocked: false,
      }
    } else {
      flowReportBuffer[fid].score = dt.score || 0
      flowReportBuffer[fid].reportText = dt.report_text || ''
    }
  }

  // 画面渲染滴答同步解锁：当画面 globalSec 推进达到 flow_end_sec 时，卡片解锁展示
  function checkUnlockReports(currentGlobalSec: number) {
    for (const fid in flowReportBuffer) {
      const r = flowReportBuffer[fid]
      if (!r.unlocked && (r.endSec <= 0 || currentGlobalSec >= r.endSec - 0.5)) {
        r.unlocked = true
        if (!completedFlows.has(fid)) {
          completedFlows.add(fid)
          segScores.value.push(r.score)
          if (r.flowType === 'supervision') supN.value++
          else if (r.flowType === 'info_notice') noticeN.value++
          else ticketN.value++
        }
        let card = segCards.value.find(c => c.flowId === fid)
        if (!card) {
          card = {
            flowId: fid,
            flowType: r.flowType,
            score: r.score,
            reportText: r.reportText,
            continueSec: r.continueSec,
            collapsed: false,
            streamBuffer: r.streamBuffer,
            streaming: !r.reportText,
          }
          segCards.value.push(card)
        }
      } else if (r.unlocked) {
        let card = segCards.value.find(c => c.flowId === fid)
        if (card) {
          card.streamBuffer = r.streamBuffer
          if (r.reportText) {
            card.score = r.score
            card.reportText = r.reportText
            card.streaming = false
          }
        }
      }
    }
  }

  function addFlow(d: any, isStart: boolean) {
    const dt = d.data || d
    const sec = Number(dt.flow_start_sec || dt.flow_end_sec || d.localSec || d.sec || 0)
    const flowType = dt.flow_type || d.flowType || 'supervision'
    const [color, name] = flowTypeMap[flowType] || ['#ffaa00', '自唱票']
    
    // 防重复推送同秒同类型事件
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
    if (label === 'voice') progress.voice = pct
    else if (label === 'tracker') progress.tracker = pct
    else if (label === 'gaze') progress.gaze = pct
    else progress.detail = dt.detail || (dt.pct !== undefined ? pct.toFixed(1) + '%' : '')
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
    frameFront, frameBup, framePop,
    hasFrameFront, hasFrameBup, hasFramePop,
  }
}
