// 从动锁步(伺服锁定模型 v5): pop 没有自己的播放进度, 每帧被主时钟(front)连续锁定
//
// 核心不变量: pop 的内容位置永远 ≤ master 前方一帧, 且随时向 masterSec 收敛——
//   超前(pop 在 master 前方): 减速/暂停等待 master 追上, 绝不向前跳(v4 实测教训:
//     media 段丢失后 buffer 只剩尾部, "前方 range 贴播"把 pop 向前 seek 到 437s,
//     超前 400s 后永远回不到 master 身边, 从动彻底失控)
//   落后(pop 在 master 后方): seek 回 masterSec; masterSec 不可达(段丢失/断档)时
//     贴播到 masterSec 前方最近 range 起点, 等 master 追上(master 在后方推进, 必然到达)
//
// 优先级链(每帧按序判定):
//   1. master 冻结(front stall) → pop 联动暂停; 恢复后跟随
//   2. 超前等待: pop 大幅超前且 masterSec 不可达 → 暂停等 master(每帧检查收敛即恢复)
//   3. 播放头安全检查: 播放头不在有效缓冲内(空洞/删头/断档) → 按超前/落后语义对齐
//   4. 暂停态: 大偏差先对齐再唤醒
//   5. 越界(master 超出 pop 存储): 贴尾随播
//   6. 硬同步(>0.3s): seek 拉回(冷却); 冷却期急伺服限幅
//   7. 伺服(>半帧): 速率微调连续收敛 ±15%
//   8. 死区(≤半帧): 完全锁定
//
// 速率控制权: pop 的 playbackRate 完全归本模块(VideoPanel watch 只管 front)
import { metrics } from './metrics'

// 内容偏移校准(秒): pop 内容 = front 内容 + 此偏移; 默认 0(两路同时开拍)
const CONTENT_OFFSET_SEC = 0
// 死区=半帧(25fps一帧40ms): 内不调, 人眼不可辨
const FRAME_DEADZONE = 0.02
// 伺服增益: 收敛时间常数 τ≈1/(base×gain)≈0.5s
const SERVO_GAIN = 2.0
// 常规伺服修正幅度上限(±15%)
const SERVO_CLAMP = 0.15
// 急伺服限幅: 追赶不吃光 buffer
const URGENT_MAX = 1.25
const URGENT_MIN = 0.75
// 硬同步阈值
const HARD_SYNC_SEC = 0.3
// seek 冷却(ms): 防 seek 风暴
const SEEK_COOLDOWN_MS = 500
// master 冻结判定帧数(0.5s @60Hz)
const MASTER_FROZEN_FRAMES = 30
// 超前暂停阈值: pop 超前 master 超过此值且 master 不可达 → 暂停等待
const AHEAD_PAUSE_SEC = 1.0

export interface SlaveSyncState {
  overEndSince: number      // masterSec 越界计时起点(0=未越界)
  lastMasterSec: number     // 上帧主时钟(冻结检测)
  frozenFrames: number      // master 连续未动帧数
  frozenPaused: boolean     // 因 master 冻结而暂停的标记
  lastSeekAt: number        // 上次发起 seek 的时刻(防风暴冷却)
  aheadPaused: boolean      // 因超前等待而暂停的标记
}

export function createSlaveSyncState(): SlaveSyncState {
  return { overEndSince: 0, lastMasterSec: -1, frozenFrames: 0, frozenPaused: false, lastSeekAt: 0, aheadPaused: false }
}

/** mediaUrl 重建(流重置)时清状态 */
export function resetSlaveSync(state: SlaveSyncState): void {
  state.overEndSince = 0
  state.lastMasterSec = -1
  state.frozenFrames = 0
  state.frozenPaused = false
  state.lastSeekAt = 0
  state.aheadPaused = false
}

function setRate(vid: HTMLVideoElement, rate: number): void {
  if (vid.playbackRate !== rate) vid.playbackRate = rate
}

/** 位置是否落在有效缓冲内(离 range 末留 0.1s 余量) */
function inSafeZone(vid: HTMLVideoElement, sec: number): boolean {
  for (let i = 0; i < vid.buffered.length; i++) {
    if (sec >= vid.buffered.start(i) && sec <= vid.buffered.end(i) - 0.1) return true
  }
  return false
}

/** 带冷却 seek: 目标须在有效缓冲内 */
function seek(vid: HTMLVideoElement, target: number, state: SlaveSyncState): boolean {
  const now = performance.now()
  if (now - state.lastSeekAt < SEEK_COOLDOWN_MS) return false
  if (!inSafeZone(vid, target)) return false
  try {
    vid.currentTime = target
    state.lastSeekAt = now
    metrics.incr('seekCount')
    return true
  } catch {
    metrics.incr('seekFailCount')
    return false
  }
}

/** 落后对齐: seek 回 masterSec; 不可达时贴播到其前方最近 range 起点(master 后方推进必然到达) */
function catchUpSeek(vid: HTMLVideoElement, target: number, state: SlaveSyncState): boolean {
  if (seek(vid, target, state)) return true
  for (let i = 0; i < vid.buffered.length; i++) {
    const s = vid.buffered.start(i)
    if (s >= target && s <= vid.buffered.end(i) - 0.1) {
      return seek(vid, s, state)
    }
  }
  return false
}

/**
 * 每帧驱动 pop 伺服锁定主时钟.
 * @param vid 从动 video(pop)
 * @param masterSec 主时钟(front currentTime)
 * @param isPlaying 应播放标记
 * @param baseRate 基准速率(与 front 同速; 伺服在其上微调)
 * @param state 锁步状态(跨帧记忆)
 */
export function syncSlave(vid: HTMLVideoElement, masterSec: number, isPlaying: boolean, baseRate: number, state: SlaveSyncState): void {
  // ── 1. master 冻结检测(front stall → pop 联动暂停, 恢复后跟随) ──
  if (Math.abs(masterSec - state.lastMasterSec) < 0.001) state.frozenFrames++
  else state.frozenFrames = 0
  state.lastMasterSec = masterSec
  if (state.frozenFrames > MASTER_FROZEN_FRAMES) {
    if (!vid.paused) { vid.pause(); state.frozenPaused = true }
    return
  }
  if (state.frozenPaused) {
    state.frozenPaused = false
    if (isPlaying && !state.aheadPaused) vid.play().catch(() => {})
  }

  const target = masterSec + CONTENT_OFFSET_SEC   // 内容校准后的锁定目标
  const e = vid.currentTime - target              // 正=pop超前, 负=pop落后
  metrics.set('slaveDrift', e)

  // ── 2. 超前等待: pop 大幅超前且 masterSec 不可达(段丢失/断档) → 暂停等 master 追 ──
  // 绝不向前跳(v4 实测教训: 向前跳导致从动彻底失控); master 后方推进必然追上
  if (e > AHEAD_PAUSE_SEC && !inSafeZone(vid, target)) {
    if (!vid.paused) { vid.pause(); state.aheadPaused = true }
    return
  }
  if (state.aheadPaused) {
    // master 追到 pop 附近(或 masterSec 变得可达) → 恢复
    if (e <= AHEAD_PAUSE_SEC || inSafeZone(vid, target)) {
      state.aheadPaused = false
      if (isPlaying) vid.play().catch(() => {})
    } else {
      return   // 继续等
    }
  }

  // ── 3. 播放头安全检查: 播放头不在有效缓冲内(空洞/删头/断档) ──
  if (vid.buffered.length > 0 && !inSafeZone(vid, vid.currentTime)) {
    if (e < 0) {
      // 落后: 对齐到 masterSec(或其前方 range 贴播)
      if (catchUpSeek(vid, target, state)) {
        setRate(vid, baseRate)
        if (vid.paused && isPlaying) vid.play().catch(() => {})
        return
      }
      // 无处可对齐: 等待供给, 停在原地
      return
    }
    // 超前且播放头脱离缓冲(极端态): 也按等待处理
    if (!vid.paused) { vid.pause(); state.aheadPaused = true }
    return
  }

  // ── 4. 暂停态(非冻结/非超前等待, 如 isPlaying 抖动竞态): 先对齐再唤醒 ──
  if (vid.paused) {
    if (isPlaying && vid.buffered.length > 0) {
      const bufEnd = vid.buffered.end(vid.buffered.length - 1)
      if (Math.abs(e) > HARD_SYNC_SEC) {
        if (e < 0 && catchUpSeek(vid, target, state)) { setRate(vid, baseRate); return }
        if (e > 0 && seek(vid, target, state)) { setRate(vid, baseRate); return }
      }
      if (bufEnd - vid.currentTime > 0.3) vid.play().catch(() => {})
    }
    return
  }

  // ── 5. 越界: master 超出 pop 存储末尾 → 贴尾随播(数据恢复即自动追上) ──
  const n = vid.buffered.length
  const bufEnd = n > 0 ? vid.buffered.end(n - 1) : 0
  if (n > 0 && target > bufEnd - 0.1) {
    const now = performance.now()
    if (!state.overEndSince) state.overEndSince = now
    else if (now - state.overEndSince > 1000) {
      const tail = bufEnd - 0.1
      if (tail > vid.currentTime) {
        try { vid.currentTime = tail; state.lastSeekAt = now; metrics.incr('seekCount') } catch { /* 忽略 */ }
      }
    }
    return
  }
  state.overEndSince = 0

  // ── 6. 硬同步: 瞬态失锁(>0.3s) → 带冷却 seek 直接拉回 ──
  if (Math.abs(e) > HARD_SYNC_SEC) {
    if (seek(vid, target, state)) {          // masterSec 可达: 无论超前落后都拉回
      setRate(vid, baseRate)
      return
    }
    if (e < 0) {
      // 落后且 masterSec 不可达: 贴播前方 range; 无 range 则急伺服追赶
      if (catchUpSeek(vid, target, state)) { setRate(vid, baseRate); return }
      const urgency = Math.max(0.5, Math.min(3.0, Math.abs(e)))
      setRate(vid, baseRate * Math.min(URGENT_MAX, 1 + urgency * SERVO_GAIN * 0.5))
      return
    }
    // 超前且 masterSec 不可达: 分支 2 已处理(暂停等待), 此处防御性减速
    setRate(vid, baseRate * URGENT_MIN)
    return
  }

  // ── 7/8. 伺服: 半帧死区内完全锁定; 否则速率微调连续收敛 ──
  if (Math.abs(e) <= FRAME_DEADZONE) {
    setRate(vid, baseRate)
    return
  }
  const factor = Math.max(1 - SERVO_CLAMP, Math.min(1 + SERVO_CLAMP, 1 - e * SERVO_GAIN))
  setRate(vid, baseRate * factor)
}
