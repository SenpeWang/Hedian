/**
 * WS 二进制帧协议解析(纯函数): [1字节 channel][1字节 type] + fMP4 段 payload.
 */
// WS 二进制帧协议解析(纯函数): [1字节 channel][1字节 type] + fMP4 段 payload
import { VIS_CHANNEL, VIS_TYPE, type ViewId, type VisSegType } from './types'

/**
 * 解析后的视频流帧: 视角 + 段类型 + fMP4 段数据.
 */
export interface VisFrame {
  /** 帧所属视角(front/pop) */
  view: ViewId
  /** 段类型(init/media/end) */
  type: VisSegType
  /** fMP4 段负载(已从协议头切出) */
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
 *
 * @param u8 - 原始 WS 二进制负载(含 2 字节协议头).
 * @returns 解析后的帧; 非法帧(长度不足/channel 未知/type 未知)返回 null.
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
 *
 * @param msg - 待判定的 JSON 消息.
 * @returns 流式报告消息返回 true.
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
 *
 * @param msg - 待判定的 JSON 消息.
 * @returns 终态报告消息返回 true.
 */
export function isTerminalMsg(msg: MsgLike): boolean {
  return msg.source === 'segment_report' || msg.type === 'report' || msg.source === 'report'
}

/**
 * 判定是否为报告类消息(流式或终态), 用于路由分流.
 *
 * @param msg - 待判定的 JSON 消息.
 * @returns 流式或终态报告消息均返回 true.
 */
export function isReportMsg(msg: MsgLike): boolean {
  return isStreamMsg(msg) || isTerminalMsg(msg)
}
