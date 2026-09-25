/**
 * 自动滚底: 监听依赖变化(nextTick 后滚到底).
 *
 * 统一 VoicePanel/ReportPanel/NotifyPanel 三处重复逻辑;
 * 使用 behavior:'auto' 即时滚动, 避免 smooth 平滑动画在高频更新(流式文本/事件涌入)下反复被打断而追不上最新.
 */
import { nextTick, watch, type Ref } from 'vue'

// 自动滚底: 监听依赖变化(nextTick 后滚到底). 统一 VoicePanel/ReportPanel/NotifyPanel 三处重复逻辑
// 使用 behavior:'auto' 即时滚动, 避免 smooth 平滑动画在高频更新(流式文本/事件涌入)下反复被打断而追不上最新
/**
 * 创建自动滚底: dep 变化后于 nextTick 把容器滚到底部.
 *
 * @param elRef - 目标滚动容器 ref.
 * @param dep - 依赖求值函数(返回值变化即触发滚底).
 * @returns 含 scrollToBottom 手动滚底方法的对象.
 */
export function useScrollBottom(elRef: Ref<HTMLElement | null>, dep: () => unknown) {
  /** 等 DOM 更新后即时滚到底部 */
  async function scrollToBottom() {
    await nextTick()
    if (elRef.value) elRef.value.scrollTo({ top: elRef.value.scrollHeight, behavior: 'auto' })
  }
  watch(dep, () => {
    scrollToBottom()
  })
  return { scrollToBottom }
}
