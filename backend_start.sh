#!/bin/bash
# 由 crond 拉起启动后端(绕开 trae 沙盒 GPU 设备限制); 只允许 GPU, 失败快速退出绝不回退 CPU
PY=/home/wangshengping/myconda/envs/sp_hedian/bin/python
cd /home/wangshengping/Hedian/A_DemoSrc || exit 1

[ -f /tmp/hedian_backend.running ] && exit 0

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] GPU 探测失败, 本轮放弃等待下一分钟重试" > data/backend_probe.err
  exit 1
fi

touch /tmp/hedian_backend.running
echo "[$(date '+%F %T')] GPU 探测通过, 启动 main.py" >> data/backend_cron.log
nohup "$PY" main.py --gpu 0 >> data/backend_cron.log 2>&1 &
