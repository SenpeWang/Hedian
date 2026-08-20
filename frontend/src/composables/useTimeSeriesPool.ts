import { reactive } from 'vue'

// 时序数据池: 按 localSec 升序. items 用 reactive, insertSorted 触发依赖 computed 重算
// (front 暂停时 currentPlaybackSec 不变, 但新数据 insertSorted 仍驱动 people/gaze computed 重算)
export class TimeSeriesPool<T extends { localSec: number }> {
  private items: T[] = reactive([]) as unknown as T[]

  clear() { this.items.splice(0, this.items.length) }
  get length() { return this.items.length }

  insertSorted(item: T) {
    const s = item.localSec
    if (this.items.length === 0 || s >= this.items[this.items.length - 1].localSec) {
      this.items.push(item); return
    }
    let lo = 0, hi = this.items.length - 1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (this.items[mid].localSec < s) lo = mid + 1
      else hi = mid - 1
    }
    this.items.splice(lo, 0, item)
  }

  // 二分找 localSec<=sec 的最后一项; 查无(时刻早于所有数据)返回 null 不返回未来项
  getLatestAt(sec: number): T | null {
    if (this.items.length === 0) return null
    let lo = 0, hi = this.items.length - 1, ans = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (this.items[mid].localSec <= sec) { ans = mid; lo = mid + 1 }
      else hi = mid - 1
    }
    return ans >= 0 ? this.items[ans] : null
  }
}
