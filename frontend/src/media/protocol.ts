// WS 二进制帧协议解析(纯函数): [1字节 channel][1字节 type] + fMP4 段 payload
import { VIS_CHANNEL, VIS_TYPE, type ViewId, type VisSegType } from './types'

/** 单帧视频流解析结果 */
export interface VisFrame {
  view: ViewId
  type: VisSegType
  data: ArrayBuffer
}

// type code → 段类型 静态反查表(模块级一次构建, 热路径 O(1) 无分配)
const TYPE_BY_CODE: Record<number, VisSegType> = {
  [VIS_TYPE.init]: 'init',
  [VIS_TYPE.media]: 'media',
  [VIS_TYPE.end]: 'end'
}

/**
 * 解析视频流二进制帧.
 * @param u8 WS 下行的二进制帧原始字节
 * @returns 解析结果; 非法帧(长度不足/channel 未知/type 未知)返回 null
 */
export function parseVisFrame(u8: Uint8Array): VisFrame | null {
  if (u8.length < 2) return null
  const view: ViewId | null =
    u8[0] === VIS_CHANNEL.front ? 'front' : u8[0] === VIS_CHANNEL.pop ? 'pop' : null
  if (!view) return null
  const type = TYPE_BY_CODE[u8[1]]
  if (!type) return null
  return { view, type, data: u8.subarray(2).slice().buffer }
}

// ── 评估报告 JSON 消息形状谓词(单一来源: 编排层路由 + 业务层分类共用, 防两份拷贝漂移) ──
type MsgLike = { source?: unknown; type?: unknown }

/**
 * 判定是否为流式 chunk 消息(segment_report_stream).
 * @param msg 任意 JSON 消息(形状未知)
 * @returns 命中流式报告消息形状时为 true
 */
export function isStreamMsg(msg: MsgLike): boolean {
  return (
    msg.source === 'segment_report_stream' ||
    msg.type === 'report_stream' ||
    msg.source === 'stream'
  )
}

/**
 * 判定是否为终态报告消息(segment_report).
 * @param msg 任意 JSON 消息(形状未知)
 * @returns 命中终态报告消息形状时为 true
 */
export function isTerminalMsg(msg: MsgLike): boolean {
  return msg.source === 'segment_report' || msg.type === 'report' || msg.source === 'report'
}

/**
 * 判定是否为报告类消息(流式或终态, 路由判定).
 * @param msg 任意 JSON 消息(形状未知)
 * @returns 命中任一报告消息形状时为 true
 */
export function isReportMsg(msg: MsgLike): boolean {
  return isStreamMsg(msg) || isTerminalMsg(msg)
}
