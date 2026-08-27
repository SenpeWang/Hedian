// 结构化分级日志: [playback][scope] 前缀, 生产可全局关 debug
const PREFIX = '[playback]'
const DEBUG_ENABLED = true   // 上线可置 false 或按 import.meta.env.DEV

export interface Logger {
  debug: (msg: string, ...args: unknown[]) => void
  info: (msg: string, ...args: unknown[]) => void
  warn: (msg: string, ...args: unknown[]) => void
  error: (msg: string, ...args: unknown[]) => void
}

export function createLogger(scope: string): Logger {
  const tag = `${PREFIX}[${scope}]`
  return {
    debug: (msg, ...args) => { if (DEBUG_ENABLED) console.log(tag, msg, ...args) },
    info: (msg, ...args) => console.log(tag, msg, ...args),
    warn: (msg, ...args) => console.warn(tag, msg, ...args),
    error: (msg, ...args) => console.error(tag, msg, ...args),
  }
}
