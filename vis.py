#!/usr/bin/env python3
"""
核电站监护制合规检测系统 — Flask Web 可视化层
路由: / /start /stream /events /audio /status
事件推送: SSE (Server-Sent Events)
"""
import os, json, threading, queue, time, copy
from flask import Flask, Response, request, jsonify, send_file

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
_pipeline_runner = None
_pipeline_thread = None
_main_module = None

HEDIAN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.abspath(os.path.join(HEDIAN, "..", "Hedian_data"))

# ── SSE 广播客户端管理 ──
_sse_clients = []          # List[queue.Queue]  每个 SSE 连接一个独立队列
_sse_clients_lock = threading.Lock()


def set_pipeline_runner(fn):
    global _pipeline_runner
    _pipeline_runner = fn


def _get_main():
    global _main_module
    if _main_module is None:
        import importlib
        _main_module = importlib.import_module("main")
    return _main_module


# ══════════════════════════════════════════════════════════
#  路由
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return _HTML_PAGE


@app.route("/start")
def start():
    global _pipeline_thread
    m = _get_main()
    if m.pipeline_state == "running":
        return jsonify({"status": "busy"}), 409
    m.pipeline_state = "running"
    _pipeline_thread = threading.Thread(target=_pipeline_runner, daemon=True)
    _pipeline_thread.start()
    return jsonify({"status": "started"})


@app.route("/events")
def events():
    """SSE 端点：每个客户端独立队列，通过广播接收事件"""
    def gen():
        client_q = queue.Queue(maxsize=256)
        with _sse_clients_lock:
            _sse_clients.append(client_q)
        try:
            while True:
                try:
                    ev = client_q.get(timeout=25)
                except queue.Empty:
                    yield ": keepalive\n\n"  # 防止超时断开
                    continue
                if ev is None:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            with _sse_clients_lock:
                if client_q in _sse_clients:
                    _sse_clients.remove(client_q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.route("/stream")
def stream():
    """MJPEG 视频流"""
    def gen():
        m = _get_main()
        while True:
            try:
                data = m.frame_queue.get(timeout=5)
            except Exception:
                continue
            if data is None:
                break
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/audio")
def audio():
    wav = os.path.join(DATA_ROOT, "data", "raw_from_video.wav")
    if os.path.exists(wav):
        return send_file(wav, mimetype="audio/wav")
    return "", 404


@app.route("/status")
def status():
    m = _get_main()
    return jsonify({"state": m.pipeline_state, "result_dir": m._result_dir})


# ══════════════════════════════════════════════════════════
#  完整 HTML 页面
# ══════════════════════════════════════════════════════════
_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>核电站行为合规检测系统</title>

<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#0a0e1a;color:#e0e6f0;overflow:hidden;height:100vh;display:flex;flex-direction:column}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0a0e1a}
::-webkit-scrollbar-thumb{background:#1e2a42;border-radius:3px}

/* ── 顶部栏 ── */
.header{display:flex;align-items:center;padding:8px 20px;background:linear-gradient(135deg,#0d1224,#141b2d);border-bottom:1px solid #1e2a42;justify-content:space-between}
.header h1{font-size:18px;color:#00d4ff;white-space:nowrap}
.header h1 span{margin-right:8px}
.header-center{display:flex;align-items:center;justify-content:center;flex:1}
#startBtn{padding:8px 28px;border:1px solid #00d4ff;border-radius:6px;background:linear-gradient(135deg,rgba(0,212,255,.1),rgba(0,100,200,.15));color:#00d4ff;cursor:pointer;font-size:15px;font-weight:600;transition:.3s;letter-spacing:1px}
#startBtn:hover{transform:scale(1.05);background:linear-gradient(135deg,rgba(0,212,255,.25),rgba(0,100,200,.3));box-shadow:0 0 12px rgba(0,212,255,.3)}
#startBtn:disabled{opacity:.5;cursor:default;transform:none;box-shadow:none}
/* ── 右上角状态区 ── */
.header-right{margin-left:auto;display:flex;align-items:center;gap:14px}
#headerStatus{font-size:13px;color:#e0e6f0;white-space:nowrap;max-width:360px;overflow:hidden;text-overflow:ellipsis}
#statusDot{width:8px;height:8px;border-radius:50%;background:#6b7a90;flex-shrink:0}
#statusDot.active{background:#00ff88;animation:pulse 1.5s infinite}

/* 进度条组 - tqdm 风格 */
.progress-group{display:flex;flex-direction:column;gap:2px;min-width:340px}
.progress-row{font-family:'Consolas','Courier New',monospace;font-size:12px;color:#8899aa;white-space:nowrap;line-height:1.4}
.progress-row .tqdm-bar{color:#00d4ff}
.progress-row .tqdm-done{color:#00ff88}
.progress-row .tqdm-error{color:#ff4d4d}

/* ── 主区 ── */
.main{display:grid;grid-template-columns:2fr 1fr 1fr;gap:8px;padding:8px;flex:7;min-height:0}
.panel{background:#141b2d;border:1px solid #1e2a42;border-radius:6px;display:flex;flex-direction:column;overflow:hidden}
.panel-title{font-size:13px;font-weight:600;padding:8px 12px;border-bottom:1px solid #1e2a42;color:#8899aa;white-space:nowrap}
.panel-body{flex:1;overflow-y:auto;padding:8px;position:relative}

/* 视频 */
#streamImg{width:100%;height:100%;object-fit:contain;display:block}
#videoOverlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:15px;color:#6b7a90;pointer-events:none}

/* 语音文字 */
.text-item{padding:4px 6px;margin-bottom:4px;border-radius:4px;font-size:12.5px;line-height:1.5;animation:fadeIn .3s ease-in}
.text-item .ts{color:#6b7a90;margin-right:6px;font-size:11px}
.intent-badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;margin-left:4px;vertical-align:middle}
.intent-监护请求,.intent-监护确认{background:rgba(255,77,77,.25);color:#ff4d4d}
.intent-确认{background:rgba(0,255,136,.2);color:#00ff88}
.intent-操作指令,.intent-操作说明,.intent-操作结束{background:rgba(0,212,255,.2);color:#00d4ff}
.intent-实验结束{background:rgba(255,215,0,.25);color:#ffd700;font-weight:700}
.kw-red{color:#ff4d4d;font-weight:700}
.kw-green{color:#00ff88;font-weight:700}
.kw-blue{color:#00d4ff;font-weight:700}

/* 行为检测 */
.det-section{margin-bottom:6px}
.det-section h4{font-size:11px;color:#6b7a90;margin-bottom:3px;padding-bottom:2px;border-bottom:1px solid #1e2a42}
.det-item{font-size:11.5px;padding:3px 6px;margin-bottom:2px;border-radius:3px;animation:fadeIn .3s ease-in}
.sup-ok{color:#00ff88}.sup-near{color:#ffaa00}.sup-far{color:#ff4d4d}

/* ── 底部报告区 ── */
.bottom{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:0 8px 8px;flex:3;min-height:0}

/* 分段评价 */
.mini-summary{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px;text-align:center}
.mini-summary div{background:#0a0e1a;border-radius:4px;padding:6px}
.mini-summary .val{font-size:20px;font-weight:700;color:#00d4ff}
.mini-summary .lbl{font-size:10px;color:#6b7a90;margin-top:2px}
.mini-card{background:#0a0e1a;border-radius:4px;padding:6px 8px;margin-bottom:4px;border-left:3px solid #6b7a90;animation:fadeIn .3s ease-in}
.mini-card .mc-head{display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px}
.mini-card .mc-score{font-weight:700}
.mini-card .mc-bar{height:4px;background:#1e2a42;border-radius:2px;margin:4px 0}
.mini-card .mc-bar-fill{height:100%;border-radius:2px;transition:width .6s ease-out}
.mini-card .mc-detail{font-size:11px;color:#8899aa;line-height:1.4}

/* 大模型报告 */
.qr-placeholder{text-align:center;padding:30px;color:#6b7a90;font-size:14px}
.qr-placeholder .icon{font-size:36px;margin-bottom:8px}
.qr-title{font-size:14px;font-weight:700;color:#ffd700;padding:6px 0 4px;border-bottom:1px solid #1e2a42;margin-bottom:4px}
.qr-score-line{text-align:center;font-size:22px;font-weight:700;padding:10px;margin:6px 0}
.qr-item{font-size:12.5px;padding:2px 0;line-height:1.6}

@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
</style>
</head>
<body>

<!-- 顶部 -->
<div class="header">
  <h1><span>⚛️</span>核电站监护制合规检测系统</h1>
  <div class="header-center">
    <button id="startBtn" onclick="startPipeline()">▶ 开始测试</button>
  </div>
  <div class="header-right">
    <div id="statusDot"></div>
    <span id="headerStatus">就绪</span>
    <div class="progress-group" id="progressGroup" style="display:none">
      <div class="progress-row" id="prVoice">🎤 语音 |░░░░░░░░░░░░░░░░░░░░|   0% 就绪</div>
      <div class="progress-row" id="prMot">🎯 跟踪 |░░░░░░░░░░░░░░░░░░░░|   0% 就绪</div>
    </div>
  </div>
</div>

<!-- 主区 三栏 -->
<div class="main">
  <!-- 视频 -->
  <div class="panel">
    <div class="panel-title">🎯 多目标跟踪</div>
    <div class="panel-body" style="padding:0">
      <img id="streamImg" alt="">
      <div id="videoOverlay">⭕ 等待视频输入</div>
    </div>
  </div>
  <!-- 语音 -->
  <div class="panel">
    <div class="panel-title">🎤 语音转文字</div>
    <div class="panel-body" id="textScroll"></div>
  </div>
  <!-- 行为 -->
  <div class="panel">
    <div class="panel-title">📊 行为检测</div>
    <div class="panel-body">
      <div class="det-section"><h4>📢 三段式沟通</h4><div id="sanduanList"></div></div>
      <div class="det-section"><h4>✋ 举手检测</h4><div id="handList"></div></div>
      <div class="det-section"><h4>👥 监护状态</h4><div id="supList"></div></div>
    </div>
  </div>
</div>

<!-- 底部报告区 -->
<div class="bottom">
  <!-- 分段评价 -->
  <div class="panel">
    <div class="panel-title">📋 分段小评价</div>
    <div class="panel-body">
      <div class="mini-summary">
        <div><div class="val" id="msCount">0</div><div class="lbl">流程数</div></div>
        <div><div class="val" id="msAvg">-</div><div class="lbl">平均分</div></div>
        <div><div class="val" id="msGrade">-</div><div class="lbl">趋势</div></div>
      </div>
      <div id="miniCards"></div>
    </div>
  </div>
  <!-- 大模型报告 -->
  <div class="panel">
    <div class="panel-title">🤖 大模型评价报告</div>
    <div class="panel-body" id="qwenPanel">
      <div class="qr-placeholder"><div class="icon">🤖</div>等待推理完成后生成…</div>
    </div>
  </div>
</div>

<audio id="audioPlayer" preload="none" style="display:none"></audio>

<script>
const $ = id => document.getElementById(id);
let evtSource = null, pipelineStartTime = 0, miniScores = [];
const audio = $('audioPlayer');

function fmt(s) { let m = Math.floor(s/60), ss = Math.floor(s%60); return `${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`; }

function startPipeline() {
  const btn = $('startBtn');
  btn.disabled = true; btn.textContent = '⏳ 正在推理中...';
  $('statusDot').className = 'active';
  $('headerStatus').textContent = '🚀 启动中…';
  // 先建立 SSE 连接，open 后再启动 pipeline
  connectSSE(function() {
    fetch('/start').then(r=>r.json()).then(d=>{
      if(d.status==='busy'){btn.textContent='⚠️ 已在运行';$('headerStatus').textContent='⚠️ 已在运行';return}
    });
  });
}

function connectSSE(onReady) {
  if (evtSource) { evtSource.close(); evtSource = null; }
  evtSource = new EventSource('/events');
  evtSource.onopen = function() {
    console.log('SSE 连接已建立');
    if (onReady) onReady();
  };
  evtSource.onmessage = function(e) {
    try {
      const d = JSON.parse(e.data);
      handleEvent(d);
      if (d.type === 'done') {
        evtSource.close(); evtSource = null;
        console.log('SSE 连接已关闭');
      }
    } catch(ex) { console.error('SSE 解析错误:', ex); }
  };
  evtSource.onerror = function() {
    console.log('SSE 连接断开');
  };
}

function scheduleDisplay(timeSec, fn) {
  if (!pipelineStartTime) { fn(); return; }
  let delay = Math.max(0, pipelineStartTime + timeSec*1000 - Date.now());
  if (delay === 0) fn(); else setTimeout(fn, delay);
}

function highlightText(t) {
  t = t.replace(/请求监护/g, '<span class="kw-red">请求监护</span>');
  t = t.replace(/(好|确认|收到|没问题)/g, '<span class="kw-green">$1</span>');
  t = t.replace(/(1ES\w+|T1RPA\w+|LCO\w+|RPA\w+|SM3)/g, '<span class="kw-blue">$1</span>');
  return t;
}

function handleEvent(d) {
  try {
  // 所有事件已由后端帧对齐释放，到达即显示
  if (d.type === 'voice') {
    addVoiceItem(d);
  } else if (d.type === 'mini_report') {
    addSanduanCard(d); addMiniReportCard(d);
  } else if (d.type === 'tracking') {
    addTrackingCard(d);
  } else if (d.type === 'supervision_report') {
    addSanduanCard(d); addMiniReportCard(d);
  } else if (d.type === 'progress') {
    updateProgress(d);
  } else if (d.type === 'status') {
    $('headerStatus').textContent = d.text;
  } else if (d.type === 'report_progress') {
    $('headerStatus').textContent = '🤖 ' + d.text;
    showReportProgress(d.text);
  } else if (d.type === 'report') {
    showQwenReport(d.text);
  } else if (d.type === 'video_start') {
    pipelineStartTime = Date.now();
    $('headerStatus').textContent = '🎯 帧对齐推理中…';
    $('videoOverlay').style.display = 'none';
    $('streamImg').src = '/stream?' + Date.now();
    audio.src = '/audio'; audio.load();
    audio.currentTime = 0; audio.play().catch(()=>{});
  } else if (d.type === 'done') {
    $('statusDot').className = '';
    $('startBtn').disabled = false;
    $('startBtn').textContent = '🔄 重新测试';
    $('headerStatus').textContent = '✅ 完成';
    $('progressGroup').style.display = 'none';
    audio.pause();
  }
  } catch(ex) { console.error('事件处理错误:', d.type, ex); }
}

function updateProgress(d) {
  $('progressGroup').style.display = '';
  let rowId;
  if (d.label === '语音处理') {
    rowId = 'prVoice';
  } else if (d.label === '目标跟踪') {
    rowId = 'prMot';
  } else if (d.label === '全局') {
    $('headerStatus').textContent = d.detail || (d.pct + '%');
    return;
  } else return;
  const icon = rowId === 'prVoice' ? '🎤 语音' : '🎯 跟踪';
  const pct = Math.max(0, Math.min(100, d.pct || 0));
  const barLen = 20;
  const filled = Math.round(pct / 100 * barLen);
  const empty = barLen - filled;
  let bar, cls;
  if (d.pct < 0) {
    bar = '█'.repeat(barLen); cls = 'tqdm-error';
  } else if (pct >= 100) {
    bar = '█'.repeat(barLen); cls = 'tqdm-done';
  } else {
    bar = '█'.repeat(filled) + '░'.repeat(empty); cls = 'tqdm-bar';
  }
  const pctStr = (pct < 10 ? '  ' : pct < 100 ? ' ' : '') + pct + '%';
  const detail = d.pct < 0 ? '✖ 错误' : (d.detail || '');
  $(rowId).innerHTML = `${icon} |<span class="${cls}">${bar}</span>|${pctStr} ${detail}`;
}

function addVoiceItem(d) {
  const el = document.createElement('div');
  el.className = 'text-item';
  let badge = d.intent ? `<span class="intent-badge intent-${d.intent}">${d.intent}</span>` : '';
  el.innerHTML = `<span class="ts">[${fmt(d.time_sec)}]</span>${highlightText(d.text||'')}${badge}`;
  $('textScroll').appendChild(el);
  el.scrollIntoView({behavior:'smooth'});
}

function addSanduanCard(d) {
  const list = $('sanduanList');
  const el = document.createElement('div');
  el.className = 'det-item';
  let scoreColor = d.score >= 8 ? '#00ff88' : d.score >= 5 ? '#ffaa00' : '#ff4d4d';
  let isSup = !!d.operator;  // 监护评价有 operator 字段
  if (isSup) {
    el.style.borderLeft = '3px solid #40e0d0';
    el.innerHTML = `<span class="ts">[${d.time||''}]</span> 🛡️ ${d.operator} <span style="color:${scoreColor};font-weight:700">${d.score}/10</span>
      <div style="font-size:11px;color:#aaa;margin-top:2px">举手:${d.hand_raise?'✅':'❌'} | 到位:${d.supervisor_arrived?'✅':'❌'} | 帧:${d.frame_id||'?'}</div>
      <div style="font-size:11px;margin-top:2px">${(d.detail||'').replace(/\|/g,' ')}</div>`;
  } else {
    el.innerHTML = `<span class="ts">[${d.time||''}]</span> ${d.device||''} <span style="color:${scoreColor};font-weight:700">${d.score}/10</span><div style="font-size:11px;color:#aaa;margin-top:2px">${(d.cmd_text||'').substring(0,60)}</div><div style="font-size:11px;margin-top:2px">${(d.detail||'').replace(/\|/g,' ')}</div>`;
  }
  list.appendChild(el);
  el.scrollIntoView({behavior:'smooth'});
}

function addTrackingCard(d) {
  const ev = d.event;
  if (ev === 'HAND_RAISE_CHECK' || ev === 'HAND_RAISE_SUPERVISION') {
    const el = document.createElement('div');
    el.className = 'det-item';
    let raised = d.raised !== false;
    let icon = raised ? '✅ 已举手' : '❌ 未举手';
    let color = raised ? '#00ff88' : '#ff4d4d';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> ${d.operator||'?'}: <span style="color:${color}">${icon}</span>`;
    $('handList').appendChild(el);
  } else if (ev === 'SUPERVISOR_STATUS') {
    const el = document.createElement('div');
    el.className = 'det-item';
    let cls = d.status==='监护中'||d.status==='到位' ? 'sup-ok' : d.status==='接近中' ? 'sup-near' : 'sup-far';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> ${d.operator||''} 距离:${d.distance_px||'?'}px <span class="${cls}"><b>${d.status||''}</b></span>`;
    const list = $('supList');
    list.insertBefore(el, list.firstChild);
    while (list.childElementCount > 10) list.removeChild(list.lastChild);
  } else if (ev === 'SUPERVISION_TRIGGERED') {
    const el = document.createElement('div');
    el.className = 'det-item';
    el.style.border = '1px solid #ff4d4d';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> ⚠️ 监护触发`;
    $('supList').insertBefore(el, $('supList').firstChild);
  } else if (ev === 'OPERATION_START') {
    const el = document.createElement('div');
    el.className = 'det-item';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> <span style="color:#00d4ff">▶ 操作开始</span>`;
    $('sanduanList').appendChild(el);
  } else if (ev === 'OPERATION_END') {
    const el = document.createElement('div');
    el.className = 'det-item';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> <span style="color:#00d4ff">■ 操作结束</span>`;
    $('sanduanList').appendChild(el);
  } else if (ev === 'EXPERIMENT_END') {
    const el = document.createElement('div');
    el.className = 'det-item';
    el.style.background = 'rgba(255,215,0,.1)';
    el.innerHTML = `<span style="color:#ffd700">🏁 实验结束</span>`;
    $('sanduanList').appendChild(el);
  } else if (ev === 'ROLE_ASSIGNED') {
    const el = document.createElement('div');
    el.className = 'det-item';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> 🎭 角色分配: ${JSON.stringify(d.details||{})}`;
    $('sanduanList').insertBefore(el, $('sanduanList').firstChild);
  } else if (ev === 'SUPERVISION_BOUND') {
    const el = document.createElement('div');
    el.className = 'det-item';
    el.style.border = '1px solid #00ff88';
    el.style.background = 'rgba(0,255,136,.08)';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> <span class="sup-ok">🔗 ${d.operator||''} 监护绑定</span>`;
    $('supList').insertBefore(el, $('supList').firstChild);
  } else if (ev === 'SUPERVISION_END') {
    const el = document.createElement('div');
    el.className = 'det-item';
    el.style.border = '1px solid #ff4d4d';
    el.style.background = 'rgba(255,77,77,.08)';
    el.innerHTML = `<span class="ts">[${fmt(d.time_sec||0)}]</span> <span class="sup-far">⚠️ 监护解绑 — ${d.reason||''}</span>`;
    $('supList').insertBefore(el, $('supList').firstChild);
  }
}

function addMiniReportCard(d) {
  miniScores.push(d.score);
  let avg = (miniScores.reduce((a,b)=>a+b,0)/miniScores.length).toFixed(1);
  $('msCount').textContent = miniScores.length;
  $('msAvg').textContent = avg;
  if (miniScores.length >= 2) {
    let last2 = miniScores.slice(-2);
    $('msGrade').textContent = last2[1]>last2[0] ? '📈 上升' : last2[1]<last2[0] ? '📉 下降' : '➡️ 持平';
  }
  $('headerStatus').textContent = `流式推理 ${miniScores.length}个流程 | 均分${avg}`;

  let isSup = !!d.operator;
  let scoreColor = d.score>=8?'#00ff88':d.score>=5?'#ffaa00':'#ff4d4d';
  let pct = d.score*10;
  let borderColor = isSup ? '#40e0d0' : scoreColor;
  let title = isSup
    ? `🛡️ 监护制度 [${d.time||''}] ${d.operator||'?'}`
    : `流程${miniScores.length} [${d.time||''}] ${d.device||''}`;
  let extra = isSup
    ? `<div style="font-size:10px;color:#888;margin-top:3px">帧:${d.frame_id||'?'} | 举手:${d.hand_raise_time_sec?fmt(d.hand_raise_time_sec):'—'} | 到位:${d.arrive_time_sec?fmt(d.arrive_time_sec):'—'} | 持续:${d.bound_duration_sec||0}s</div>`
    : '';

  const card = document.createElement('div');
  card.className = 'mini-card';
  card.style.borderLeftColor = borderColor;
  card.innerHTML = `
    <div class="mc-head">
      <span>${title}</span>
      <span class="mc-score" style="color:${scoreColor}">${d.score}/10</span>
    </div>
    <div class="mc-bar"><div class="mc-bar-fill" style="width:${pct}%;background:${scoreColor}"></div></div>
    <div class="mc-detail">${(d.detail||'').replace(/\|/g,'<br>')}</div>
    ${extra}
  `;
  $('miniCards').appendChild(card);
  card.scrollIntoView({behavior:'smooth'});
}

function showReportProgress(text) {
  $('qwenPanel').innerHTML = `<div class="qr-placeholder" style="animation:pulse 1.5s infinite"><div class="icon">🤖</div>${text}</div>`;
}

function showQwenReport(text) {
  const panel = $('qwenPanel');
  panel.innerHTML = '';
  const lines = text.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    let m = line.match(/总分[：:]\s*(\d+)\/100\s+评级[：:]\s*([ABCD])/);
    if (m) {
      let color = m[2]==='A'?'#00ff88':m[2]==='B'?'#00d4ff':m[2]==='C'?'#ffaa00':'#ff4d4d';
      const el = document.createElement('div');
      el.className = 'qr-score-line';
      el.style.color = color;
      el.textContent = line.trim();
      panel.appendChild(el);
      $('headerStatus').textContent = '✅ ' + line.trim();
      continue;
    }
    if (/^(#{1,3}\s|[一二三四五六七八九十]、)/.test(line.trim())) {
      const el = document.createElement('div');
      el.className = 'qr-title';
      el.textContent = line.replace(/^#+\s*/, '').trim();
      panel.appendChild(el); continue;
    }
    const el = document.createElement('div');
    el.className = 'qr-item';
    let html = line.replace(/✅/g,'<span style="color:#00ff88">✅</span>')
                   .replace(/❌/g,'<span style="color:#ff4d4d">❌</span>')
                   .replace(/⚠️/g,'<span style="color:#ffaa00">⚠️</span>');
    el.innerHTML = html;
    panel.appendChild(el);
  }
}

// 页面加载时自动连接，保持长连接不断开

</script>
</body>
</html>
"""
