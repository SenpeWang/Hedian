// 播放编排核心: 组合 media 内核(WsClient/MediaBuffer/RateController/RetentionCenter/Scheduler)
// 职责: 管线状态机 / 主时钟 / 媒体 URL / 速率 / 消息路由(内核→业务 store 回调注入)
// 消息流单向: WS → 内核 → usePlayback 路由 → 业务 store(useTranscript/useNotify/useReports)
import { ref, computed, onBeforeUnmount, type Ref } from 'vue'
import { createMediaBuffer, type MediaBuffer } from '../media/mse-buffer'
import { RateController } from '../media/rate-controller'
import { Scheduler } from '../media/scheduler'
import { createWsClient, type WsClient } from '../media/ws-client'
import { isReportMsg } from '../media/protocol'
import { createLogger } from '../media/logger'
import { metrics } from '../media/metrics'
import { PLAYBACK, STATUS_TRANSITIONS, type PipelineStatus, type ViewId } from '../media/types'
import { startPipeline as apiStart, stopPipeline as apiStop } from '../api/pipeline'

const log = createLogger('playback')

export function usePlayback(currentPlaybackSec: Ref<number>) {
  // ── 播放状态(对外 reactive) ──
  const status = ref<PipelineStatus>('idle')
  const isPlaying = computed(() => status.value === 'running' || status.value === 'starting')
  const totalDuration = ref(0)
  const globalSec = ref(0)                 // 后端推理进度(进度条用, 超前播放)
  const viewSecs = ref({ front: 0, pop: 0, voice: 0 })
  const playbackRate = ref(1.0)
  const frontMediaUrl = ref('')
  const popMediaUrl = ref('')
  // 各路供给终止标记: front/pop 收到 end 段即完成; voice 用进度阈值推断完成
  const visEnded: Record<ViewId, boolean> = { front: false, pop: false }

  // ── 内核组合 ──
  const rateController = new RateController()
  const scheduler = new Scheduler(250)
  let frontBuf: MediaBuffer | null = null
  let popBuf: MediaBuffer | null = null
  const _visInitSeen: Record<ViewId, boolean> = { front: false, pop: false }   // init 每路只入队一次

  // 延迟注册的业务回调(App.vue 创建业务 store 后注入, 破解循环依赖)
  let batchHandler: ((raw: Record<string, unknown>) => void) | null = null
  let reportHandler: ((msg: Record<string, unknown>) => void) | null = null

  // ── 管线状态机(集中赋值, 非法转换 dev 告警) ──
  function setStatus(next: PipelineStatus): void {
    const from = status.value
    if (from === next) return
    if (!STATUS_TRANSITIONS[from].includes(next)) {
      log.warn(`非法状态转换: ${from} → ${next}(允许: ${STATUS_TRANSITIONS[from].join(',')})`)
    }
    status.value = next
  }

  // ── MSE 生命周期(重建前显式销毁旧 buffer, 修泄漏) ──
  function initMSE(): void {
    frontBuf?.destroy(); popBuf?.destroy()
    if (frontMediaUrl.value) { try { URL.revokeObjectURL(frontMediaUrl.value) } catch { /* 忽略 */ } }
    if (popMediaUrl.value) { try { URL.revokeObjectURL(popMediaUrl.value) } catch { /* 忽略 */ } }
    // front:含音频轨;pop:仅视频
    const frontMS = new MediaSource()
    frontMediaUrl.value = URL.createObjectURL(frontMS)
    frontBuf = createMediaBuffer({
      mediaSource: frontMS, codec: 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"',
      view: 'front', getClock: () => currentPlaybackSec.value, logger: log,
    })
    const popMS = new MediaSource()
    popMediaUrl.value = URL.createObjectURL(popMS)
    popBuf = createMediaBuffer({
      mediaSource: popMS, codec: 'video/mp4; codecs="avc1.42E01E"',
      view: 'pop', getClock: () => currentPlaybackSec.value, logger: log,
    })
  }

  // ── 视频帧路由(init 去重/end 置完成/media 入 buffer) ──
  function onVisFrame(frame: { view: ViewId; type: 'init' | 'media' | 'end'; data: ArrayBuffer }): void {
    const buf = frame.view === 'front' ? frontBuf : popBuf
    if (!buf) {
      // 加载时序窗口(MSE 未建)丢段告警: media 段丢了不补发(时间轴从此断档)
      log.error(`no buffer for ${frame.view}, 丢弃 ${frame.type} 段(后端不补发 media!)`)
      return
    }
    if (frame.type === 'init') {
      // init 段去重: 重复 append 解码配置会破坏 buffer(重连时后端会补发)
      if (_visInitSeen[frame.view]) return
      _visInitSeen[frame.view] = true
      log.info(`INIT ${frame.view} len=${frame.data.byteLength}`)
      buf.push(frame.data)
      return
    }
    if (frame.type === 'end') {
      log.info(`END ${frame.view}(供给终止, 移出限速)`)
      visEnded[frame.view] = true
      buf.end()
      return
    }
    buf.push(frame.data)
  }

  // ── JSON 消息路由: status/done 本层消化; report/batch 回调业务层 ──
  function onJson(msg: Record<string, unknown>): void {
    if (!msg) return
    const source = msg.source as string | undefined
    // 连接时后端补发的状态快照: 稳定 totalDuration(不依赖 video 加载)+当前 globalSec
    if (source === 'status') {
      const total = msg.totalDuration as number | undefined
      const gs = msg.globalSec as number | undefined
      const st = msg.status as PipelineStatus | undefined
      if (total && total > 0) totalDuration.value = total
      if (gs != null && gs > 0) globalSec.value = gs
      if (st) setStatus(st)
      return
    }
    if (source === 'done' || (msg.meta && (msg.meta as { source?: string }).source === 'done')) {
      setStatus('done')
      // 推理完成: 进度满(不依赖最后 batch 的 globalSec 是否到 total)
      if (totalDuration.value > 0) globalSec.value = totalDuration.value
      return
    }
    if (isReportMsg(msg)) { reportHandler?.(msg); return }
    // batch 事件(评估/flow/progress/语音/人数/凝视): globalSec/sourceTimes 本层消化, 其余回调
    if (msg.globalSec != null) globalSec.value = msg.globalSec as number
    if (msg.totalDuration && (msg.totalDuration as number) > 0) totalDuration.value = msg.totalDuration as number
    const st = msg.sourceTimes as Record<string, number> | undefined
    if (st) viewSecs.value = {
      front: st.front ?? viewSecs.value.front,
      pop: st.pop ?? viewSecs.value.pop,
      voice: st.voice ?? viewSecs.value.voice,
    }
    batchHandler?.(msg)
  }

  // ── WS 客户端 ──
  const wsClient: WsClient = createWsClient({
    url: () => `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/data`,
    onVisFrame,
    onJson,
  })

  // ── 水位速率 tick: 采水位 + 供给速率差分 → RateController(完成路移出限速) ──
  // 供给估计: 各未完成路供给位(bufferEnd/viewSec)的差分+当前播放速率 = 供给速率,
  // 取 min(瓶颈路) EMA 平滑; 用作速率恢复上限, 恒定慢供给下稳定贴供给走(防 0.2x↔1.0x 振荡)
  let _supplyLast: { front: number; pop: number; voice: number } | null = null
  let _supplyLastTime = 0
  let _supplyEma = 1.0

  // ── JSON 进度路死源判定: 连续停滞且落后播放头的源(如模块崩溃后进度恒 0)移出限速 ──
  // 若不剔除, 其 level 持续深负会把整体速率压死在 RATE_MIN, 后端供完了前端还在爬行
  const _progSeen: Record<string, number> = {}
  const _stallSince: Record<string, number> = {}
  function pathAlive(key: string, sec: number, clock: number, nowMs: number): boolean {
    const seen = _progSeen[key]
    if (seen !== undefined && sec <= seen) {
      if (!_stallSince[key]) _stallSince[key] = nowMs
      else if (nowMs - _stallSince[key] > PLAYBACK.SOURCE_STALL_MS && sec < clock - 3) return false
    } else {
      _progSeen[key] = sec
      delete _stallSince[key]
    }
    return true
  }

  function rateTick(): void {
    // LEAD 节流期 append 不产生 updateend, 由 tick 兜底驱动 pump 重试(防空转断链)
    frontBuf?.kick()
    popBuf?.kick()
    const t = currentPlaybackSec.value
    const now = performance.now()
    if (t > 0) {
      const voiceDone = totalDuration.value > 0 && viewSecs.value.voice >= totalDuration.value - 5
      const levels: number[] = []
      if (frontBuf && !visEnded.front) levels.push(frontBuf.getBufferedEnd() - t)
      if (popBuf && !visEnded.pop) levels.push(popBuf.getBufferedEnd() - t)
      if (!voiceDone && pathAlive('voice', viewSecs.value.voice, t, now)) {
        levels.push(viewSecs.value.voice - t)
      }
      // 供给速率采样: 只用视频路 bufferEnd(连续平滑), 不用 voice(批量推送跳变会污染估计)
      const ends = {
        front: frontBuf && !visEnded.front ? frontBuf.getBufferedEnd() : null,
        pop: popBuf && !visEnded.pop ? popBuf.getBufferedEnd() : null,
        voice: null,
      } as Record<string, number | null>
      if (!_supplyLast) {
        _supplyLast = { front: ends.front ?? 0, pop: ends.pop ?? 0, voice: ends.voice ?? 0 }
        _supplyLastTime = now
      } else if (now - _supplyLastTime > 1000) {
        const dt = (now - _supplyLastTime) / 1000
        const rates: number[] = []
        for (const k of ['front', 'pop', 'voice'] as const) {
          const e = ends[k]
          if (e != null && e > 0) {
            // 供给速率 = 水位增速 + 播放消耗(播放暂停时退化为水位增速, 保守方向)
            rates.push(Math.max(0, (e - _supplyLast[k]) / dt + playbackRate.value))
          }
        }
        if (rates.length) {
          const bottleneck = Math.min(1.5, Math.min(...rates))   // clamp 上界防 voice 批量跳变
          _supplyEma = _supplyEma * 0.7 + bottleneck * 0.3
        }
        _supplyLast = { front: ends.front ?? 0, pop: ends.pop ?? 0, voice: ends.voice ?? 0 }
        _supplyLastTime = now
      }
      playbackRate.value = rateController.update(levels, playbackRate.value, _supplyEma)
    }
    // metrics 快照
    metrics.set('clock', currentPlaybackSec.value)
    metrics.set('rate', playbackRate.value)
    metrics.set('bufferLevel', {
      front: frontBuf ? Math.max(0, frontBuf.getBufferedEnd() - currentPlaybackSec.value) : 0,
      pop: popBuf ? Math.max(0, popBuf.getBufferedEnd() - currentPlaybackSec.value) : 0,
    })
    metrics.set('queueDepth', {
      front: frontBuf?.getQueueDepth() ?? 0,
      pop: popBuf?.getQueueDepth() ?? 0,
    })
    metrics.flush()
  }

  scheduler.every(PLAYBACK.RATE_TICK_MS, rateTick)

  // ── 主时钟上报(供后端评估 wait_playback_reached) ──
  let lastReportSec = 0
  let lastReportReal = 0
  function reportPlaybackProgress(currentSec: number): void {
    currentPlaybackSec.value = currentSec
    const now = performance.now()
    if (Math.abs(currentSec - lastReportSec) < 0.5 && now - lastReportReal < 500) return
    lastReportSec = currentSec; lastReportReal = now
    wsClient.sendReport(currentSec)
  }

  // ── 进度条 = globalSec/totalDuration(推理进度, 超前播放); 仅 done 强制 100 ──
  const progress = computed(() =>
    status.value === 'done' ? 100
      : (totalDuration.value > 0 ? Math.min(100, globalSec.value / totalDuration.value * 100) : 0))

  function fmt(s?: number): string {
    if (s === undefined || s === null || isNaN(s)) return '00:00'
    const m = Math.floor(s / 60), sec = Math.floor(s % 60)
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }

  // ── 全量重置(重新推理) ──
  function resetState(): void {
    totalDuration.value = 0
    globalSec.value = 0
    playbackRate.value = 1.0
    viewSecs.value = { front: 0, pop: 0, voice: 0 }
    visEnded.front = false; visEnded.pop = false
    _visInitSeen.front = false; _visInitSeen.pop = false
    // 死源停滞观测表复位(新一轮推理的源重新获得限速资格)
    for (const k of Object.keys(_progSeen)) delete _progSeen[k]
    for (const k of Object.keys(_stallSince)) delete _stallSince[k]
    currentPlaybackSec.value = 0
    lastReportSec = 0; lastReportReal = 0
    metrics.reset()
    initMSE()   // 重建 MSE(新 objectURL, 清空 buffer)
  }

  async function startPipeline(): Promise<void> {
    // 调用方(App.handleStart)已先 resetAll 全量重置(含 MSE 重建), 此处只负责状态机+API
    setStatus('starting')
    try {
      await apiStart()
      setStatus('running')
    } catch (err) {
      log.error('启动失败:', err)
      setStatus('idle')
    }
  }

  async function stopPipeline(): Promise<void> {
    try { await apiStop() }
    catch (e) { log.error('停止请求失败:', e) }
    setStatus('stopped')   // 用户停止不等于推理完成, 不跳 100%
  }

  // ── 业务层延迟注册(App.vue 创建 store 后调用) ──
  function onBatch(fn: (raw: Record<string, unknown>) => void): void { batchHandler = fn }
  function onReport(fn: (msg: Record<string, unknown>) => void): void { reportHandler = fn }

  // ── 生命周期: 启动连接与调度; 卸载全清理 ──
  // setup 即建 MSE: WS connect 先于 App onMounted 的 resetAll, 不预建则加载时序窗口内
  // 到达的 media 段被丢弃(后端不补发, 时间轴断档——实测教训); resetAll 会重建一次, 代价可忽略
  initMSE()
  wsClient.connect()
  scheduler.start()
  onBeforeUnmount(() => {
    scheduler.stop()
    wsClient.destroy()
    frontBuf?.destroy(); popBuf?.destroy()
    if (frontMediaUrl.value) { try { URL.revokeObjectURL(frontMediaUrl.value) } catch { /* 忽略 */ } }
    if (popMediaUrl.value) { try { URL.revokeObjectURL(popMediaUrl.value) } catch { /* 忽略 */ } }
  })

  return {
    // 状态
    status, isPlaying, progress, fmt,
    currentPlaybackSec, playbackRate, totalDuration, globalSec,
    frontMediaUrl, popMediaUrl,
    // 控制
    startPipeline, stopPipeline, resetState, reportPlaybackProgress,
    // 业务层注册
    onBatch, onReport,
  }
}
