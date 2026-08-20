export type FlowType = 'supervision' | 'self_ticket' | 'info_notice'

export interface VoiceEntry {
  sec: number
  text: string
  keys: string[]
}

export interface FlowEvent {
  sec: number
  flowType: FlowType
  name: string
  color: string
  isStart: boolean
}

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
