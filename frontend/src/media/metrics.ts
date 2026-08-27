// 运行时指标: 计数器 + 最新值快照, 每次快照刷新到 window.__playbackMetrics(devtools 排查, 不外发)
export interface PlaybackMetrics {
  // 累计计数器
  appendFail: number          // appendBuffer 非 Quota 异常次数
  quotaErrors: number         // QuotaExceededError 次数(自愈环触发)
  droppedSegments: number     // 丢段数(队满/重试超限; 正常应为 0)
  reportCopiesReleased: number // 报告全文副本释放数
  reconnects: number          // WS 重连次数
  seekCount: number           // 伺服层 seek 次数(自愈/硬同步; 稳态应趋于 0 增长)
  seekFailCount: number       // seek 失败次数(应恒 0)
  // 最新值快照(每次 flush 刷新)
  clock: number               // 主时钟(源视频秒)
  rate: number                // 当前播放速率
  slaveDrift: number          // pop 相对主时钟偏差(秒, 正=超前; 收敛后应 <0.04 一帧)
  bufferLevel: { front: number; pop: number }  // 各路水位(bufferEnd - clock)
  queueDepth: { front: number; pop: number }   // 各路 append 队列深度
}

declare global {
  interface Window { __playbackMetrics?: PlaybackMetrics }
}

class MetricsStore {
  private data: PlaybackMetrics = {
    appendFail: 0, quotaErrors: 0, droppedSegments: 0,
    reportCopiesReleased: 0, reconnects: 0,
    seekCount: 0, seekFailCount: 0,
    clock: 0, rate: 1.0, slaveDrift: 0,
    bufferLevel: { front: 0, pop: 0 },
    queueDepth: { front: 0, pop: 0 },
  }

  incr<K extends keyof PlaybackMetrics>(key: K, n = 1): void {
    const v = this.data[key]
    if (typeof v === 'number') (this.data[key] as number) = v + n
  }

  set<K extends keyof PlaybackMetrics>(key: K, value: PlaybackMetrics[K]): void {
    this.data[key] = value
  }

  /** 快照刷到 window.__playbackMetrics(调度周期调用) */
  flush(): void {
    try { window.__playbackMetrics = { ...this.data, bufferLevel: { ...this.data.bufferLevel }, queueDepth: { ...this.data.queueDepth } } } catch { /* SSR/隐私模式忽略 */ }
  }

  reset(): void {
    this.data.appendFail = 0; this.data.quotaErrors = 0; this.data.droppedSegments = 0
    this.data.reportCopiesReleased = 0; this.data.reconnects = 0
    this.data.seekCount = 0; this.data.seekFailCount = 0
    this.data.clock = 0; this.data.rate = 1.0; this.data.slaveDrift = 0
    this.data.bufferLevel = { front: 0, pop: 0 }
    this.data.queueDepth = { front: 0, pop: 0 }
    this.flush()
  }
}

export const metrics = new MetricsStore()
