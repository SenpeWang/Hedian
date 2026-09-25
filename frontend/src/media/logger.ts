/**
 * 结构化分级日志: [playback][scope] 前缀, 生产可全局关 debug.
 */
// 结构化分级日志: [playback][scope] 前缀, 生产可全局关 debug
const PREFIX = '[playback]'
const DEBUG_ENABLED = true // 上线可置 false 或按 import.meta.env.DEV

/**
 * 分级日志接口(按 scope 打标签, 可注入以便替换/静音).
 */
export interface Logger {
  /** 调试日志(受 DEBUG_ENABLED 开关控制) */
  debug: (msg: string, ...args: unknown[]) => void
  /** 常规信息日志 */
  info: (msg: string, ...args: unknown[]) => void
  /** 告警日志(异常但可自愈) */
  warn: (msg: string, ...args: unknown[]) => void
  /** 错误日志(需人工关注) */
  error: (msg: string, ...args: unknown[]) => void
}

/**
 * 创建带 scope 前缀的分级日志器.
 *
 * @param scope - 模块标识, 拼接为 `[playback][scope]` 前缀.
 * @returns 该 scope 下的 Logger 实例.
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
