import { ref, reactive } from 'vue'

export interface ProgressState {
  voice: number
  tracker: number
  gaze: number
  detail: string
}

export function useSSE() {
  const status = ref<'idle' | 'starting' | 'running' | 'done'>('idle')
  const statusText = ref('就绪')
  let dataSource: EventSource | null = null
  let isFirstBatch = true

  // ── Single-Tick 联合原子响应式渲染变量 ──
  const frameFront = ref('')
  const frameBup = ref('')
  const framePop = ref('')
  const hasFrameFront = ref(false)
  const hasFrameBup = ref(false)
  const hasFramePop = ref(false)

  // ── 数据列表 ──
  const voiceEntries = ref<any[]>([])
  const voiceMap = reactive<Record<number, any>>({})
  const people = reactive({ count: '--', alert: '就绪', alertColor: '#8899aa' })
  const gaze = reactive({ hasHeads: false, headsCount: 0, anyInRoi: false, awayDuration: 0 })
  const flowEvents = ref<any[]>([])
  const segCards = ref<any[]>([])
  const segScores = ref<number[]>([])
  const supN = ref(0), ticketN = ref(0), noticeN = ref(0)
  const completedFlows = new Set<string>()
  const progress = reactive<ProgressState>({ voice: 0, tracker: 0, gaze: 0, detail: '' })

  function fmt(sec: any) {
    if (sec === undefined || sec === null) return '00:00'
    const s = Math.floor(Number(sec))
    const m = Math.floor(s / 60)
    const r = s % 60
    return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0')
  }

  function connect(onConnected?: () => void) {
    if (dataSource) dataSource.close()
    dataSource = new EventSource('/data')
    dataSource.onopen = () => {
      onConnected?.()
    }
    dataSource.onmessage = (evt) => {
      try {
        const d = JSON.parse(evt.data)
        handleData(d)
      } catch (ex) { console.error('SSE处理异常:', ex) }
    }
    dataSource.onerror = () => {
      status.value = 'idle'
      statusText.value = '就绪'
    }
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
    frameFront.value = ''; frameBup.value = ''; framePop.value = ''
    hasFrameFront.value = false; hasFrameBup.value = false; hasFramePop.value = false
    voiceEntries.value = []
    for (const k in voiceMap) delete voiceMap[Number(k)]
    flowEvents.value = []
    segCards.value = []
    segScores.value = []
    supN.value = 0; ticketN.value = 0; noticeN.value = 0
    completedFlows.clear()
    people.count = '--'; people.alert = '就绪'; people.alertColor = '#8899aa'
    gaze.hasHeads = false; gaze.headsCount = 0; gaze.anyInRoi = false; gaze.awayDuration = 0
    progress.voice = 0; progress.tracker = 0; progress.gaze = 0; progress.detail = ''
  }

  // ── Single-Tick 联合 Batch 同步解包渲染 ──
  function handleData(d: any) {
    if (d.globalSec !== undefined) {
      const batchSec = Number(d.globalSec)

      // 1. 首包流式 Batch 到达调起音频播放，防止声音抢跑
      if (isFirstBatch) {
        isFirstBatch = false
        const audioEl = document.getElementById('mainAudio') as HTMLAudioElement
        if (audioEl) {
          audioEl.src = '/api/audio/stream?t=' + Date.now()
          audioEl.currentTime = batchSec || 0
          audioEl.play().catch(e => console.warn('Audio play deferred:', e))
        }
      }

      // 2. 5 秒滑动窗口倍速软调节
      adjustAudioSpeed(batchSec)

      // 3. Single-Tick 联合同步刷新三视角 Base64
      const nextFront = d.video_front?.[0]?.data?.frame_data
        ? 'data:image/jpeg;base64,' + d.video_front[0].data.frame_data
        : frameFront.value

      const nextBup = d.video_bup?.[0]?.data?.frame_data
        ? 'data:image/jpeg;base64,' + d.video_bup[0].data.frame_data
        : frameBup.value

      const nextPop = d.video_pop?.[0]?.data?.frame_data
        ? 'data:image/jpeg;base64,' + d.video_pop[0].data.frame_data
        : framePop.value

      frameFront.value = nextFront; hasFrameFront.value = !!nextFront
      frameBup.value = nextBup; hasFrameBup.value = !!nextBup
      framePop.value = nextPop; hasFramePop.value = !!nextPop

      // 4. 流式挂载 Gaze 凝视数据
      if (d.gaze?.[0]?.data) {
        const dt = d.gaze[0].data
        gaze.hasHeads = !!dt.has_heads
        gaze.headsCount = dt.heads_count || 0
        gaze.anyInRoi = !!dt.any_in_roi
        gaze.awayDuration = dt.away_duration || 0
      }

      // 5. 流式挂载 Voice 语音转录数据
      if (d.voice && Array.isArray(d.voice)) {
        for (const v of d.voice) addVoice(v)
      }

      // 6. 流式挂载 Tracker 目标跟踪数据
      if (d.tracker && Array.isArray(d.tracker)) {
        for (const t of d.tracker) addTracking(t)
      }

      // 7. 流式挂载流程事件
      if (d.flow_start && Array.isArray(d.flow_start)) {
        for (const f of d.flow_start) addFlow(f, true)
      }
      if (d.flow_end && Array.isArray(d.flow_end)) {
        for (const f of d.flow_end) addFlow(f, false)
      }
      return
    }
    dispatchItem(d)
  }

  function adjustAudioSpeed(globalSec: number) {
    const audioEl = document.getElementById('mainAudio') as HTMLAudioElement
    if (!audioEl || audioEl.paused) return
    const audioSec = audioEl.currentTime
    const delta = globalSec - audioSec
    // 5 秒滑动窗口 [globalSec - 2.5s, globalSec + 2.5s]
    if (Math.abs(delta) <= 2.5) {
      audioEl.playbackRate = 1.0
    } else if (delta > 2.5) {
      audioEl.playbackRate = 1.15
    } else {
      audioEl.playbackRate = 0.85
    }
  }

  function dispatchItem(d: any) {
    const source = d.source
    if (source === 'voice') addVoice(d)
    else if (source === 'tracking') addTracking(d)
    else if (source === 'gaze') addGaze(d)
    else if (source === 'progress') updateProgress(d)
    else if (source === 'flow_start') addFlow(d, true)
    else if (source === 'flow_end') addFlow(d, false)
    else if (source === 'segment_report') addSegReport(d)
    else if (source === 'done') {
      status.value = 'done'
      statusText.value = '✅ 完成'
    }
  }

  function addVoice(d: any) {
    const dt = d.data || {}
    const sec = d.localSec || dt.sec || 0
    if (!voiceMap[sec]) {
      voiceMap[sec] = { sec, text: dt.text || '', keys: dt.keys || [] }
      voiceEntries.value.push(voiceMap[sec])
      if (voiceEntries.value.length > 200) voiceEntries.value.shift()
    } else if (dt.text) {
      voiceMap[sec].text = dt.text
      if (dt.keys) voiceMap[sec].keys = dt.keys
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

  function addGaze(d: any) {
    const dt = d.data || d
    gaze.hasHeads = !!dt.has_heads
    gaze.headsCount = dt.heads_count || 0
    gaze.anyInRoi = !!dt.any_in_roi
    gaze.awayDuration = dt.away_duration || 0
  }

  const flowTypeMap: Record<string, [string, string]> = {
    supervision: ['#00d4ff', '监护制'],
    info_notice: ['#00ffcc', '信息通报'],
  }
  function addFlow(d: any, isStart: boolean) {
    const dt = d.data || d
    const [color, name] = flowTypeMap[dt.flow_type] || ['#ffaa00', '自唱票']
    flowEvents.value.push({
      sec: isStart ? dt.flow_start_sec : dt.flow_end_sec,
      flowType: dt.flow_type,
      name, color, isStart,
    })
  }

  function addSegReport(d: any) {
    const dt = d.data || d
    const fid = dt.flow_id || dt.flowId
    if (!fid) return
    completedFlows.add(String(fid))
    segScores.value.push(dt.score || 0)
    if (dt.flow_type === 'supervision') supN.value++
    else if (dt.flow_type === 'info_notice') noticeN.value++
    else ticketN.value++

    let card = segCards.value.find(c => c.flowId === String(fid))
    if (!card) {
      card = { flowId: String(fid), flowType: dt.flow_type, score: 0, reportText: '', continueSec: 0, collapsed: false, streamBuffer: '', streaming: false }
      segCards.value.push(card)
    }
    card.score = dt.score || 0
    card.reportText = dt.report_text || ''
    card.continueSec = dt.flow_continue_sec || '?'
    card.streaming = false
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

  return {
    status, statusText, startPipeline, fmt,
    voiceEntries, people, gaze, flowEvents,
    segCards, segScores, supN, ticketN, noticeN,
    progress, totalCount, avgScore,
    frameFront, frameBup, framePop,
    hasFrameFront, hasFrameBup, hasFramePop,
  }
}
