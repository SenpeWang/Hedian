<script setup lang="ts">
defineProps<{
  title: string
  frameSrc: string
  hasFrame: boolean
}>()
</script>

<template>
  <div class="panel" style="flex:1">
    <div class="panel-title">{{ title }}</div>
    <div class="panel-body" style="padding:0;flex:1;position:relative">
      <img v-if="frameSrc" :src="frameSrc" class="stream-img" alt="">
      <div v-if="!hasFrame" class="video-placeholder">
        <div class="icon">🎥</div>
        <div class="text">等待视频输入</div>
      </div>
    </div>
  </div>
</template>
