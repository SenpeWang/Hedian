// MediaBuffer: MSE SourceBuffer 封装(状态机 + append 队列 + trim + 背压 + EOS 宽限)
import { PLAYBACK, type ViewId } from './types'
import type { Logger } from './logger'
import { metrics } from './metrics'

const EOS_GRACE_MS = 3000   // end 后宽限, 期内新数据到达则取消收尾

type BufferState = 'created' | 'attaching' | 'ready' | 'closed'

export interface MediaBufferDeps {
  mediaSource: MediaSource
  codec: string
  view: ViewId
  getClock: () => number
  logger: Logger
}

export interface MediaBuffer {
  push: (data: ArrayBuffer) => void
  end: () => void
  kick: () => void
  getBufferedEnd: () => number
  getQueueDepth: () => number
  destroy: () => void
}

export function createMediaBuffer(deps: MediaBufferDeps): MediaBuffer {
  const { mediaSource: ms, codec, view, getClock, logger } = deps
  const log = (level: 'debug' | 'info' | 'warn' | 'error', msg: string, ...args: unknown[]) =>
    logger[level](`[${view}] ${msg}`, ...args)

  let sb: SourceBuffer | null = null
  let state: BufferState = 'created'
  const queue: ArrayBuffer[] = []
  let pending = false
  let ended = false
  let endedWanted = false
  let eosTimer: ReturnType<typeof setTimeout> | null = null
  let trimming = false
  let failCount = 0

  const onSourceOpen = () => {
    if (state !== 'attaching') return
    try {
      sb = ms.addSourceBuffer(codec)
      sb.mode = 'segments'   // 时戳归位
      state = 'ready'
      log('info', `addSourceBuffer OK ${codec} (segments)`)
      sb.addEventListener('updateend', onUpdateEnd)
      sb.addEventListener('error', (e) => log('error', 'SourceBuffer error', e))
      pump()
    } catch (e) {
      state = 'closed'
      log('error', 'addSourceBuffer 失败:', e)
    }
  }

  const triggerEOS = () => {
    if (eosTimer) { clearTimeout(eosTimer); eosTimer = null }
    eosTimer = setTimeout(() => {
      eosTimer = null; ended = true; state = 'closed'
      try { if (ms.readyState === 'open') ms.endOfStream() } catch { /* 忽略 */ }
    }, EOS_GRACE_MS)
    log('info', `EOS 宽限 ${EOS_GRACE_MS}ms 启动`)
  }

  const cancelEOS = () => {
    if (eosTimer) { clearTimeout(eosTimer); eosTimer = null; endedWanted = false }
    log('info', 'EOS 宽限取消(新媒体段到达)')
  }

  const onUpdateEnd = () => {
    if (state !== 'ready') return
    pending = false
    trimming ? (trimming = false, pump(), maybeTrim()) : (maybeTrim(), pump())
    if (endedWanted && !ended && queue.length === 0 && !pending && sb && !sb.updating) {
      triggerEOS()
    }
  }

  function tryRemove(start: number, end: number): boolean {
    if (!sb || sb.updating || end - start < 0.01) return false
    trimming = true
    try { sb.remove(start, end); return true } catch { trimming = false; return false }
  }

  // 常规维护只做头窗删已播 [*, t-8]。
  // 尾窗(未播内容)不可在常规 tick 删除: 视频流是一次性流式供给(Redis Stream 即时转发),
  // 供给快于播放时(如 pop ~1.7x vs RATE_MAX=1.0x), 超前段 append 后立即被删即永久丢失,
  // 播放头追进空洞 → 该视角画面冻死。尾裁仅在配额吃紧时执行(forceTrimTail)。
  function maybeTrim() {
    if (!sb || sb.updating || state !== 'ready') return
    const t = getClock()
    if (t < 10) return
    const n = sb.buffered.length
    if (n === 0) return
    const cutHead = t - PLAYBACK.TRIM_HEAD_KEEP_SEC
    for (let i = 0; i < n; i++) {
      const s = sb.buffered.start(i), e = sb.buffered.end(i)
      if (e <= cutHead) { if (tryRemove(s, e)) return; continue }
      if (s < cutHead) { if (tryRemove(s, cutHead)) return; continue }
      break
    }
  }

  // 配额兜底: 连头带尾一起裁(保留 TAIL 预读), 只在 appendBuffer 报配额满时调用
  function forceTrimTail() {
    if (!sb || sb.updating || state !== 'ready') return
    const t = getClock()
    const n = sb.buffered.length
    if (n === 0) return
    const cutTail = t + PLAYBACK.TRIM_TAIL_KEEP_SEC
    for (let i = 0; i < n; i++) {
      const s = sb.buffered.start(i), e = sb.buffered.end(i)
      if (s >= cutTail) { if (tryRemove(s, e)) return; continue }
      if (e > cutTail) { if (tryRemove(cutTail, e)) return; continue }
    }
  }

  function pump() {
    if (pending || !sb || sb.updating || queue.length === 0 || state !== 'ready') return
    // append 超前控制: 始终执行(endedWanted 排空也不豁免)——否则 end 段后队列一口气
    // 全量 appendBuffer 会爆 SourceBuffer 配额(实测 QuotaExceeded 25 万次风暴),
    // 触发 forceTrimTail 在时间轴上裁出空洞, 从动视角 seek 落洞后减速死等 → 画面停死。
    // 始终仅当 SB 超前播放头不足 LEAD 才出队, 分批排空, 播到尾自然 flush 触发 EOS。
    // 节流期无 updateend 事件, 由 rateTick 的 kick() 兜底驱动重试
    {
      const bufEnd = sb.buffered.length ? sb.buffered.end(sb.buffered.length - 1) : 0
      if (bufEnd - getClock() > PLAYBACK.APPEND_LEAD_SEC) return
    }
    const buf = queue[0]
    pending = true
    try {
      sb.appendBuffer(buf)
      queue.shift()
      failCount = 0
    } catch (e: unknown) {
      pending = false
      const name = (e as { name?: string })?.name
      if (name === 'QuotaExceededError') {
        metrics.incr('quotaErrors')
        log('warn', 'appendBuffer 配额满, 主动 trim')
        forceTrimTail()
        return
      }
      metrics.incr('appendFail')
      failCount++
      log('warn', 'appendBuffer 失败:', e)
      if (failCount > 3) { queue.shift(); failCount = 0; metrics.incr('droppedSegments') }
    }
  }

  ms.addEventListener('sourceopen', onSourceOpen)
  state = 'attaching'

  return {
    push: (data: ArrayBuffer) => {
      if (state === 'closed') {
        // EOS 宽限已触发后又收到新媒体段(供给端存在 end 提早落流的间歇现象):
        // MSE 规范允许 ended 态 appendBuffer 自动回到 open, 据此复活接收;
        // 否则该视角此后全部段被静默丢弃 → 播放头追进空洞 → 画面冻死
        if (!sb || ms.readyState === 'closed') return
        log('warn', 'EOS 已触发后又收到媒体段, 复活接收')
        state = 'ready'
        ended = false
        endedWanted = false
      }
      if (eosTimer) cancelEOS()
      queue.push(data); pump()
    },
    end: () => {
      if (ended || endedWanted) return
      if (eosTimer) { clearTimeout(eosTimer); eosTimer = null }
      endedWanted = true
      // LEAD 节流可能在队列非空时暂停过出队, 这里主动驱动排空(队空则直接触发 EOS)
      pump()
      if (queue.length === 0 && !pending && sb && !sb.updating) triggerEOS()
    },
    getBufferedEnd: () => sb && sb.buffered.length > 0 ? sb.buffered.end(sb.buffered.length - 1) : 0,
    getQueueDepth: () => queue.length,
    kick: () => { pump() },
    destroy: () => {
      state = 'closed'; queue.length = 0
      if (eosTimer) { clearTimeout(eosTimer); eosTimer = null }
      ms.removeEventListener('sourceopen', onSourceOpen)
      if (sb) try { sb.removeEventListener('updateend', onUpdateEnd) } catch { }
      try { if (ms.readyState === 'open') ms.endOfStream() } catch { }
    },
  }
}