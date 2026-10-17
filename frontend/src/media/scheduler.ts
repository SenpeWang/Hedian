/**
 * 统一调度器: 单一 interval 分频驱动所有周期任务(消除散落 setInterval 的时序交织).
 */
export class Scheduler {
  private timerId: ReturnType<typeof setInterval> | null = null
  private tickCount = 0
  private readonly tasks: { everyNTicks: number; fn: () => void }[] = []

  constructor(private readonly baseIntervalMs = 250) {}

  /**
   * 注册周期任务(ms 会被归一到 baseIntervalMs 的整数倍).
   * @param ms 期望执行周期(毫秒)
   * @param fn 任务函数(异常被捕获上报, 不影响其他任务)
   */
  every(ms: number, fn: () => void): void {
    const n = Math.max(1, Math.round(ms / this.baseIntervalMs))
    this.tasks.push({ everyNTicks: n, fn })
  }

  /** 启动调度(幂等, 已在跑则忽略) */
  start(): void {
    if (this.timerId !== null) return
    this.tickCount = 0
    this.timerId = setInterval(() => {
      this.tickCount++
      for (const t of this.tasks) {
        if (this.tickCount % t.everyNTicks === 0) {
          try {
            t.fn()
          } catch (e) {
            console.error('[playback][scheduler] task error', e)
          }
        }
      }
    }, this.baseIntervalMs)
  }

  /** 停止调度(未启动时为空操作) */
  stop(): void {
    if (this.timerId !== null) {
      clearInterval(this.timerId)
      this.timerId = null
    }
  }
}
