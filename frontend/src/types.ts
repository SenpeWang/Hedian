export interface BatchEvent {
  localSec: number
  tag: string
  data: Record<string, any>
  source?: string
}

export interface WSBatch {
  globalSec: number
  video_front?: BatchEvent[]
  video_pop?: BatchEvent[]
  voice?: BatchEvent[]
  tracking?: BatchEvent[]
  gaze?: BatchEvent[]
  flow_start?: BatchEvent[]
  flow_end?: BatchEvent[]
  progress?: BatchEvent[]
}

export interface VoiceEntry {
  sec: number
  text: string
  keys: string[]
}

export interface FlowEvent {
  sec: number
  flowType: string
  name: string
  color: string
  isStart: boolean
}

export interface SegCard {
  flowId: string
  flowType: string
  score: number
  reportText: string
  continueSec: number | string
  collapsed: boolean
  streamBuffer: string
  streaming: boolean
}

export interface ProgressState {
  voice: number
  tracker: number
  gaze: number
  detail: string
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
