/**
 * HTTP API 客户端: 管线控制接口封装(超时 + 错误规范化).
 */
// HTTP API 客户端: 管线控制接口封装(超时 + 错误规范化)
import { createLogger } from '../media/logger'

const log = createLogger('api')

/**
 * 管线 API 错误: 统一携带出错 endpoint 与原始 cause.
 */
export class PlaybackApiError extends Error {
  /**
   * 构造 API 错误, 归一携带出错 endpoint 与原始 cause.
   *
   * @param message - 错误描述文本.
   * @param endpoint - 出错的接口路径.
   * @param cause - 原始异常(可选).
   */
  constructor(
    message: string,
    public readonly endpoint: string,
    public readonly cause?: unknown
  ) {
    super(message)
    this.name = 'PlaybackApiError'
  }
}

async function post(endpoint: string, timeoutMs = 10000): Promise<void> {
  try {
    const res = await fetch(endpoint, { method: 'POST', signal: AbortSignal.timeout(timeoutMs) })
    if (!res.ok) throw new PlaybackApiError(`HTTP ${res.status}`, endpoint)
  } catch (e) {
    if (e instanceof PlaybackApiError) {
      log.error(`${endpoint} 失败`, e.message)
      throw e
    }
    const err = new PlaybackApiError(String((e as Error)?.message || e), endpoint, e)
    log.error(`${endpoint} 请求异常`, err.message)
    throw err
  }
}

/**
 * 启动推理管线.
 *
 * @returns 请求完成的 Promise; 失败抛 PlaybackApiError.
 */
export function startPipeline(): Promise<void> {
  return post('/start')
}

/**
 * 停止推理管线.
 *
 * @returns 请求完成的 Promise; 失败抛 PlaybackApiError.
 */
export function stopPipeline(): Promise<void> {
  return post('/stop')
}

/**
 * 页面刷新重置(kill 子进程+清 Redis, 回干净 idle).
 *
 * @returns 请求完成的 Promise; 失败抛 PlaybackApiError.
 */
export function resetPipeline(): Promise<void> {
  return post('/reset')
}
