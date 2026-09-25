/**
 * 统一调度器: 单一 interval 分频驱动所有周期任务(消除散落 setInterval 的时序交织).
 */
// 统一调度器: 单一 interval 分频驱动所有周期任务(消除散落 setInterval 的时序交织)
export class Scheduler {
  private timerId: ReturnType<typeof setInterval> | null = null
  private tickCount = 0
  private readonly tasks: { everyNTicks: number; fn: () => void }[] = []

  /**
   * 构造调度器并指定基础 tick 间隔.
   *
   * @param baseIntervalMs - 基础 tick 间隔(其余任务周期归一到它的整数倍).
   */
  constructor(private readonly baseIntervalMs = 250) {}

  /**
   * 注册周期任务(ms 会被归一到 baseIntervalMs 的整数倍).
   *
   * @param ms - 期望执行周期(毫秒).
   * @param fn - 到点执行的回调(单次异常不影响其他任务).
   */
  every(ms: number, fn: () => void): void {
    const n = Math.max(1, Math.round(ms / this.baseIntervalMs))
    this.tasks.push({ everyNTicks: n, fn })
  }

  /** 启动调度(已启动则忽略), tick 计数归零 */
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

  /** 停止调度并清空定时器(未启动则忽略) */
  stop(): void {
    if (this.timerId !== null) {
      clearInterval(this.timerId)
      this.timerId = null
    }
  }
}
