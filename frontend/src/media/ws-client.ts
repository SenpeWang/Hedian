// WS 客户端: 指数退避重连 + 有界播放点上报缓冲 + 消息分发(二进制/文本双通道)
// 语义与重构前一致(重连退避/onopen flush), 增加: 有界缓冲(丢旧留新)+显式销毁+重连计数
import { parseVisFrame, type VisFrame } from './protocol'
import { PLAYBACK } from './types'
import { createLogger } from './logger'
import { metrics } from './metrics'

const log = createLogger('ws')

export interface WsClientDeps {
  url: () => string
  onVisFrame: (frame: VisFrame) => void
  onJson: (msg: Record<string, unknown>) => void
}

export interface WsClient {
  connect: () => void
  /** 播放点上报(供后端 wait_playback_reached); 非 OPEN 时入有界缓冲 */
  sendReport: (sec: number) => void
  destroy: () => void
}

export function createWsClient(deps: WsClientDeps): WsClient {
  const { url, onVisFrame, onJson } = deps
  let socket: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0
  let destroyed = false
  const pendingReports: number[] = []   // 断线暂存(有界, 丢旧留新)

  function scheduleReconnect() {
    if (destroyed || reconnectTimer) return
    reconnectAttempts++
    metrics.incr('reconnects')
    const delay = Math.min(30000, 1000 * 2 ** Math.min(reconnectAttempts, 5))   // 指数退避, 上限 30s
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connect() }, delay)
  }

  function connect() {
    if (destroyed) return
    if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) return
    try { socket = new WebSocket(url()) } catch { scheduleReconnect(); return }
    socket.binaryType = 'arraybuffer'

    socket.onopen = () => {
      reconnectAttempts = 0
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
      // flush 重连期间暂存的上报(只保留最新值即可)
      while (pendingReports.length && socket) {
        const s = pendingReports.shift()!
        try { socket.send(JSON.stringify({ type: 'playback_progress', current_sec: Number(s.toFixed(2)) })) }
        catch { pendingReports.unshift(s); break }
      }
    }
    socket.onmessage = (evt) => {
      try {
        if (evt.data instanceof ArrayBuffer) {
          const frame = parseVisFrame(new Uint8Array(evt.data))
          if (frame) onVisFrame(frame)
        } else if (typeof evt.data === 'string') {
          onJson(JSON.parse(evt.data))
        }
      } catch (err) { log.warn('处理消息失败:', err) }
    }
    socket.onerror = () => { /* onclose 统一处理重连 */ }
    socket.onclose = () => { socket = null; scheduleReconnect() }
  }

  return {
    connect,
    sendReport: (sec: number) => {
      const payload = JSON.stringify({ type: 'playback_progress', current_sec: Number(sec.toFixed(2)) })
      if (socket && socket.readyState === WebSocket.OPEN) {
        try { socket.send(payload) }
        catch { pendingReports.push(sec) }
      } else {
        // 有界: 播放点上报只需最新值, 丢旧留新
        pendingReports.push(sec)
        if (pendingReports.length > PLAYBACK.MAX_PENDING_REPORTS) pendingReports.shift()
      }
    },
    destroy: () => {
      destroyed = true
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
      if (socket) {
        socket.onopen = null; socket.onmessage = null
        socket.onerror = null; socket.onclose = null   // 摘监听防僵尸回调
        try { socket.close() } catch { /* 忽略 */ }
        socket = null
      }
      pendingReports.length = 0
    },
  }
}
