/**
 * 业务数据模型: 流程/字幕/通知/报告卡片的共享类型定义.
 */

/** 流程类型: 监护制 / 自唱票 / 信息通报 */
export type FlowType = 'supervision' | 'self_ticket' | 'info_notice'

/** 一条语音字幕条目 */
export interface VoiceEntry {
  /** 归并键兼显示时刻(源视频秒) */
  sec: number
  /** 识别文本 */
  text: string
  /** 命中关键词列表(供高亮渲染) */
  keys: string[]
}

/** 一条流程事件(开始或结束) */
export interface FlowEvent {
  /** 事件时刻(源视频秒) */
  sec: number
  /** 流程类型 */
  flowType: FlowType
  /** 中文流程名(渲染用) */
  name: string
  /** 主题色(渲染用) */
  color: string
  /** true=流程开始, false=流程结束 */
  isStart: boolean
}

/** 评估报告卡片(流式累积 → 终态) */
export interface SegCard {
  /** 流程 ID(唯一键) */
  flowId: string
  /** 流程类型 */
  flowType: FlowType
  /** 评分 0-10(终态才有效) */
  score: number
  /** 终态报告全文(流式期为空, 由打字机揭示) */
  reportText: string
  /** 流程时长(秒; 后端也可能给字符串, 原样透传) */
  continueSec: number | string
  /** 折叠态(由 useReports 持有修改权) */
  collapsed: boolean
  /** 流式累积缓冲(打字机的数据源) */
  streamBuffer: string
  /** 是否仍在流式输出(打字机追完即置 false) */
  streaming: boolean
}

/** 凝视状态(状态量, 只存最新值) */
export interface GazeState {
  /** 是否检测到人头 */
  hasHeads: boolean
  /** 检测到的人头数 */
  headsCount: number
  /** 是否有人落在 ROI 内 */
  anyInRoi: boolean
  /** 无人注视 ROI 的持续时长(秒) */
  awayDuration: number
}

/** 监控室人数状态(状态量, 只存最新值) */
export interface PeopleState {
  /** 当前人数(未知时为 '--') */
  count: number | string
  /** 状态提示文案 */
  alert: string
  /** 状态配色(与 alert 联动) */
  alertColor: string
}
