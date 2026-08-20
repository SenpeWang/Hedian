import { nextTick, watch, type Ref } from 'vue'

// 自动滚底: 监听依赖变化(nextTick 后滚到底). 统一 VoicePanel/ReportPanel/NotifyPanel 三处重复逻辑
export function useScrollBottom(elRef: Ref<HTMLElement | null>, dep: () => unknown) {
  async function scrollToBottom() {
    await nextTick()
    if (elRef.value) elRef.value.scrollTop = elRef.value.scrollHeight
  }
  watch(dep, () => { scrollToBottom() })
  return { scrollToBottom }
}
