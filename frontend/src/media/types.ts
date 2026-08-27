// 播放内核类型与常量(收口所有可调参数; 语义与重构前一致)
export const PLAYBACK = {
  TRIM_HEAD_KEEP_SEC: 8,          // 视频头窗: 已播回看余量
  TRIM_TAIL_KEEP_SEC: 180,        // forceTrimTail 兜底窗(仅 QuotaExceeded 时调用)
  APPEND_LEAD_SEC: 90,            // append 超前窗口: SB 常驻数据超过播放头此后秒数即暂停出队.
  // 流式一次性供给不可再生, SB 囤积全片会撞配额(QutoaExceeded 自愈会删掉
  // 未来段→永久空洞), 队列丢弃更会直接制造"已收未播"区间的洞;
  // 故把超前段滞留在内存队列(顺序 FIFO 不丢), 靠时间推进逐步放行
  MAX_PENDING_REPORTS: 100,       // 断线播放点上报缓冲上限(丢旧留新, 上报只需最新值)
  RATE_TICK_MS: 500,              // 水位速率决策周期
  SOURCE_STALL_MS: 20000,         // JSON 进度路连续停滞阈值(超时且落后播放头→视为死源移出限速)
  RATE_MIN: 0.2,
  RATE_MAX: 1.0,
} as const

export type ViewId = 'front' | 'pop'

export type PipelineStatus = 'idle' | 'starting' | 'running' | 'done' | 'stopped'

// 管线状态机合法转换表(非法转换 dev 告警, 防状态悄悄漂移)
// idle→running/done: 页面刷新时后端仍在跑/已完成, 从 status 快照恢复的场景
export const STATUS_TRANSITIONS: Record<PipelineStatus, PipelineStatus[]> = {
  idle: ['starting', 'stopped', 'running', 'done'],
  starting: ['running', 'idle', 'stopped'],
  running: ['done', 'stopped', 'idle'],
  done: ['starting', 'idle', 'stopped'],
  stopped: ['starting', 'idle'],
}

// WS 二进制帧协议: [1字节 channel][1字节 type] + payload
export const VIS_CHANNEL = { front: 0, pop: 1 } as const
export const VIS_TYPE = { init: 0, media: 1, end: 2 } as const
export type VisSegType = 'init' | 'media' | 'end'
