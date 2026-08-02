#!/usr/bin/env bash
# 前端构建脚本
# 系统 node 过旧（不支持可选链 ?. / top-level await），vue-tsc 与 vite 均会失败；
# 必须使用 sp_hedian conda 环境的 Node 20 构建。
set -e
export PATH="/home/wangshengping/myconda/envs/sp_hedian/bin:$PATH"
cd "$(dirname "$0")"
echo "Node: $(node --version)"
npm run build
