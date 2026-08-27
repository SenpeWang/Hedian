// 水位速率引擎(纯逻辑, 可单测): 防断供的直接信号是各路"存储余量"(水位), 非增速统计
// v3 贴供给模型(实测教训: 共享 GPU 供给剧烈波动 0.5x~7x):
//   速率贴供给估计走 → 供给降速率提前跟降(不撞缓冲墙, 音视频同流不卡顿), 供给升稳定回 1.0
// 音频质量优先(变速 artifacts 是爆破音主因): 重EMA + 0.05量化 + 死区, 速率长时间稳定在少数档
import { PLAYBACK } from './types'

// 速率量化步长: 速率只在 0.05 档位间跳变(减少变速次数 → 减少音频时间拉伸 artifacts)
const RATE_STEP = 0.05
// 变化死区: EMA 后与当前差不足一档则不动(防微振荡)
const RATE_DEADBAND = RATE_STEP

export class RateController {
  /**
   * @param levels 有效水位数组(未完成路的 bufferEnd/viewSec - 主时钟); 空数组=全部完成
   * @param currentRate 当前速率(EMA 基准)
   * @param supplyEstimate 供给速率估计(媒体秒/真实秒, 编排层差分采样; 默认 1.0=未知不限制)
   * @returns 新速率(clamp [RATE_MIN, RATE_MAX], 量化 0.05 档)
   */
  update(levels: number[], currentRate: number, supplyEstimate = 1.0): number {
    let target: number
    if (levels.length === 0) {
      target = 1.0                                        // 全部完成: 正常速度播完余量
    } else {
      const minLevel = Math.min(...levels)
      if (minLevel < 1) {
        target = currentRate * 0.7                        // 撞墙紧急刹车(水位见底)
      } else {
        target = Math.min(PLAYBACK.RATE_MAX, Math.max(PLAYBACK.RATE_MIN, supplyEstimate))  // 贴供给走
        if (minLevel < 3) target = Math.min(target, currentRate)   // 低水位只降不升(防撞墙)
      }
    }
    target = Math.min(PLAYBACK.RATE_MAX, Math.max(PLAYBACK.RATE_MIN, target))
    // 重 EMA(0.75/0.25): 变化平缓; 量化到 0.05 档 + 死区: 无变化时绝不微调(音频零扰动)
    const next = Math.round((currentRate * 0.75 + target * 0.25) / RATE_STEP) * RATE_STEP
    if (Math.abs(next - currentRate) < RATE_DEADBAND) return currentRate
    return Math.min(PLAYBACK.RATE_MAX, Math.max(PLAYBACK.RATE_MIN, next))
  }
}
