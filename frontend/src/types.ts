/** 流程类型(规程状态机四类流程, 与后端 flow_type 一致) */
export type FlowType = 'supervision' | 'self_ticket' | 'info_notice' | 'normal_comm'

/** 单条语音字幕条目(keys 为标红关键词) */
export interface VoiceEntry {
  sec: number
  text: string
  keys: string[]
}

/** 系统通知里的一条流程开始/结束事件 */
// 行为检测事件(举手 / 手指指向屏幕 / 手指指向文件)
export interface BehaviorEvent {
  sec: number
  tag: string
  name: string
  color: string
}

// 通知栏事件流的一条: 流程事件与行为检测事件合并后的统一渲染形状
export interface NotifyEvent {
  sec: number
  text: string
  color: string
  // 去重与同秒稳定排序键
  key: string
}

export interface FlowEvent {
  sec: number
  flowType: FlowType
  name: string
  color: string
  isStart: boolean
  // 规则层的流程编号（同一 flowType 内自增）：仅用于"同一秒内多条流程通知"
  // 的稳定排序，见 useNotify.addFlow
  flowId?: number
}

/** 评估报告卡片(流式累积 + 完成态) */
export interface SegCard {
  flowId: string
  flowType: FlowType
  score: number
  reportText: string
  continueSec: number | string
  collapsed: boolean
  streamBuffer: string
  streaming: boolean
}

/** 凝视状态量(监控室人员关注情况) */
export interface GazeState {
  hasHeads: boolean
  headsCount: number
  anyInRoi: boolean
  awayDuration: number
}

/** 人员状态量(监控室人数与告警) */
export interface PeopleState {
  count: number | string
  alert: string
  alertColor: string
}
