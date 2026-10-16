//: unknown = 后端未给 flow_type / 给了未登记的制度名；只用于中性展示，绝不等于"监护制"
export type FlowType = 'supervision' | 'info_notice' | 'unknown'

//: 通知行类型（同刻排序用）: start=流程开始, step=子步骤判定, end=流程结束
export type FlowKind = 'start' | 'step' | 'end'

export interface VoiceEntry {
  sec: number
  text: string
  keys: string[]
}

export interface FlowEvent {
  sec: number
  flowType: FlowType
  color: string
  //: 整行展示文案（流程行为"信息通报开始"，步骤行为"未举手"/"回复请讲"）
  text: string
  //: true=制度层实时下发的子步骤判定行，false=流程开始/结束行
  isStep: boolean
  //: 步骤判定结果；流程行恒为 null
  ok: boolean | null
  //: 行类型，用于同一时刻内的稳定排序（开始 → 子步骤 → 结束）
  kind: FlowKind
  //: 同类流程内的序号（第 1/2/… 个），供系统通知区分同一视频中的多个同类型流程；0=未知
  flowSeq: number
}

export interface SegCard {
  // 卡片唯一主键 = `${flowType}:${flowId}`。各制度的 flow_id 都从 1 独立自增，
  // 只拿 flow_id 当主键时 info_notice#1 与 supervision#1 会撞车、互相覆盖
  flowKey: string
  // 该制度内的序号，仅用于展示（"信息通报 #1" / "监护制 #1"）
  flowId: string
  // 流式首帧可能尚未带 flow_type（''），终态报告到达时回填，避免误标成"监护制"
  flowType: FlowType | ''
  score: number
  reportText: string
  continueSec: number | string
  // 流程真实时间区间（秒），后端 segment_report 下发：用于按开始时刻排序与显示区间
  startSec?: number | null
  endSec?: number | null
  collapsed: boolean
  streamBuffer: string
  streaming: boolean
}

export interface GazeState {
  hasHeads: boolean
  headsCount: number
  anyInRoi: boolean
  awayDuration: number
}

export interface PeopleState {
  count: number | string
  alert: string
  alertColor: string
}
