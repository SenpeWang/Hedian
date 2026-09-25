<script setup lang="ts">
/**
 * VoicePanel: 人员对话记录面板(关键词高亮 + 自动滚底).
 *
 * 按主时钟过滤后的字幕条目逐行展示, 命中关键词以 kw-red 样式高亮.
 */
import { ref } from 'vue'
import type { VoiceEntry } from '../types'
import { useScrollBottom } from '../composables/useScrollBottom'

const props = defineProps<{
  entries: VoiceEntry[]
  fmt: (s: number) => string
}>()

const scrollEl = ref<HTMLElement | null>(null)
// 滚底: 行数 + 单句文本实时增长
useScrollBottom(
  scrollEl,
  () => props.entries.length + '|' + props.entries.map((e) => e.text).join('\n')
)

function escapeHtml(s: string) {
  return s.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!
  )
}

function renderText(entry: VoiceEntry) {
  if (!entry.text) return ''
  if (!entry.keys.length) return escapeHtml(entry.text)
  const ranges: [number, number][] = []
  for (const h of entry.keys) {
    let idx = entry.text.indexOf(h)
    while (idx !== -1) {
      ranges.push([idx, idx + h.length])
      idx = entry.text.indexOf(h, idx + 1)
    }
  }
  if (!ranges.length) return escapeHtml(entry.text)
  ranges.sort((a, b) => a[0] - b[0] || b[1] - b[0] - (a[1] - a[0]))
  let out = '',
    last = 0
  for (const r of ranges) {
    if (r[0] < last) continue
    out +=
      escapeHtml(entry.text.slice(last, r[0])) +
      `<span class="kw-red">${escapeHtml(entry.text.slice(r[0], r[1]))}</span>`
    last = r[1]
  }
  out += escapeHtml(entry.text.slice(last))
  return out
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">🎤 人员对话记录</div>
    <div ref="scrollEl" class="panel-body">
      <div v-for="entry in entries" :key="entry.sec" class="text-item">
        <span class="ts">[{{ fmt(entry.sec) }}]</span>
        <span v-html="renderText(entry)" />
      </div>
    </div>
  </div>
</template>
