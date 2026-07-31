<script setup lang="ts">
import { nextTick, watch, ref } from 'vue'
import type { VoiceEntry } from '../types'

const props = defineProps<{
  entries: VoiceEntry[]
  fmt: (s: number) => string
}>()

const scrollEl = ref<HTMLElement | null>(null)

watch(() => props.entries.length, async () => {
  await nextTick()
  if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight
})

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!))
}

function renderText(entry: VoiceEntry) {
  if (!entry.text) return ''
  if (!entry.keys.length) return escapeHtml(entry.text)
  let ranges: [number, number][] = []
  for (const h of entry.keys) {
    let idx = entry.text.indexOf(h)
    while (idx !== -1) { ranges.push([idx, idx + h.length]); idx = entry.text.indexOf(h, idx + 1) }
  }
  if (!ranges.length) return escapeHtml(entry.text)
  ranges.sort((a, b) => a[0] - b[0] || (b[1] - b[0]) - (a[1] - a[0]))
  let out = '', last = 0
  for (const r of ranges) {
    if (r[0] < last) continue
    out += escapeHtml(entry.text.slice(last, r[0])) + `<span class="kw-red">${escapeHtml(entry.text.slice(r[0], r[1]))}</span>`
    last = r[1]
  }
  out += escapeHtml(entry.text.slice(last))
  return out
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">🎤 人员对话记录</div>
    <div class="panel-body" ref="scrollEl">
      <div v-for="entry in entries" :key="entry.sec" class="text-item">
        <span class="ts">[{{ fmt(entry.sec) }}]</span>
        <span v-html="renderText(entry)"></span>
      </div>
    </div>
  </div>
</template>
