// 结构化分级日志: [playback][scope] 前缀, 生产可全局关 debug
const PREFIX = '[playback]'
const DEBUG_ENABLED = true // 上线可置 false 或按 import.meta.env.DEV

/** 分级日志接口(播放内核统一日志出口) */
export interface Logger {
  debug: (msg: string, ...args: unknown[]) => void
  info: (msg: string, ...args: unknown[]) => void
  warn: (msg: string, ...args: unknown[]) => void
  error: (msg: string, ...args: unknown[]) => void
}

/**
 * 创建带作用域前缀的分级日志器.
 * @param scope 作用域标识(如 'ws'/'api'), 展示为 `[playback][scope]` 前缀
 * @returns 四级日志方法集合(debug 受 DEBUG_ENABLED 开关控制)
 */
export function createLogger(scope: string): Logger {
  const tag = `${PREFIX}[${scope}]`
  return {
    debug: (msg, ...args) => {
      if (DEBUG_ENABLED) console.log(tag, msg, ...args)
    },
    info: (msg, ...args) => console.log(tag, msg, ...args),
    warn: (msg, ...args) => console.warn(tag, msg, ...args),
    error: (msg, ...args) => console.error(tag, msg, ...args)
  }
}
