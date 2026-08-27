// HTTP API 客户端: 管线控制接口封装(超时 + 错误规范化)
import { createLogger } from '../media/logger'

const log = createLogger('api')

export class PlaybackApiError extends Error {
  constructor(message: string, public readonly endpoint: string, public readonly cause?: unknown) {
    super(message)
    this.name = 'PlaybackApiError'
  }
}

async function post(endpoint: string, timeoutMs = 10000): Promise<void> {
  try {
    const res = await fetch(endpoint, { method: 'POST', signal: AbortSignal.timeout(timeoutMs) })
    if (!res.ok) throw new PlaybackApiError(`HTTP ${res.status}`, endpoint)
  } catch (e) {
    if (e instanceof PlaybackApiError) { log.error(`${endpoint} 失败`, e.message); throw e }
    const err = new PlaybackApiError(String((e as Error)?.message || e), endpoint, e)
    log.error(`${endpoint} 请求异常`, err.message)
    throw err
  }
}

/** 启动推理管线 */
export function startPipeline(): Promise<void> {
  return post('/start')
}

/** 停止推理管线 */
export function stopPipeline(): Promise<void> {
  return post('/stop')
}

/** 页面刷新重置(kill 子进程+清 Redis, 回干净 idle) */
export function resetPipeline(): Promise<void> {
  return post('/reset')
}
