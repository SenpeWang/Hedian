/**
 * 保留策略中心(播完释放): 报告副本释放.
 *
 * 视频媒体窗口由 MediaBuffer.maybeTrim 自治(updateend 驱动, 未播存储[t,t+60]/播完释放[t-8,t]);
 * 状态量(人数/凝视)实时显示只存最新值(常数内存), 无需驱逐.
 */
// 保留策略中心(播完释放): 报告副本释放
// 视频媒体窗口由 MediaBuffer.maybeTrim 自治(updateend 驱动, 未播存储[t,t+60]/播完释放[t-8,t])
// 状态量(人数/凝视)实时显示只存最新值(常数内存), 无需驱逐
import { metrics } from './metrics'

/**
 * 保留策略容器: 在完成态释放与主文本等价的大块副本, 压内存峰值.
 */
export class RetentionCenter {
  /**
   * 报告副本释放: 完成态后 streamBuffer 与 reportText 等价, 置空(显示 fallback reportText).
   *
   * @param card - 目标报告卡片, 仅置空其 streamBuffer 字段.
   */
  releaseReportCopy(card: { streamBuffer: string }): void {
    if (card.streamBuffer) {
      card.streamBuffer = ''
      metrics.incr('reportCopiesReleased')
    }
  }
}

// 模块级单例(无状态策略对象, usePlayback/useReports 共享)
/** 模块级单例(无状态策略对象, usePlayback/useReports 共享) */
export const retentionCenter = new RetentionCenter()
