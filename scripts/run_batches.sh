#!/usr/bin/env bash
# 自动分批向量化：每批用全新 Python 进程跑，跑完自动开下一批。
# 断点续跑：fast_embed.py 只处理 vector_status=0，每批自动接着上批的进度。
#
# 关键：限制 BLAS/torch 线程数。系统提交内存紧张时，默认 8 线程会分配大量
# 缓冲区导致 OpenBLAS 内存分配失败 → Segfault(139)。限制后内存大降、速度仍够。
#
# 用法：bash scripts/run_batches.sh
# 停止：直接关掉终端，或用任务管理停掉本脚本；已入库数据不丢。

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY=".venv/Scripts/python.exe"
BATCH=2000
LOG="$ROOT/.embed_batches.log"

# 限制线程数，降低内存占用（必须在启动 python 前 export）
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== 分批向量化启动 $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"

i=0
while true; do
  i=$((i+1))
  echo "--- 第 $i 批开始 $(date '+%H:%M:%S')，本批上限 $BATCH 条 ---" >> "$LOG"

  PYTHONPATH=".venv/Lib/site-packages;." "$PY" "scripts/fast_embed.py" --limit "$BATCH" >> "$LOG" 2>&1
  rc=$?

  if [ "$rc" -ne 0 ]; then
    echo "!!! 第 $i 批异常退出(rc=$rc) $(date '+%H:%M:%S')，5 分钟后重试同批 !!!" >> "$LOG"
    echo "!!! 详见 $LOG !!!"
    sleep 300
    continue
  fi

  # 直接查数据库判断是否还有剩余
  remaining=$("$PY" -c "
import pymysql
from dotenv import dotenv_values
cfg = dotenv_values('.env')
conn = pymysql.connect(host='127.0.0.1', user='root', password=cfg['DB_PASSWORD'], database='tcm', charset='utf8mb4')
with conn.cursor() as c:
    c.execute('SELECT COUNT(*) FROM knowledge WHERE review_status=1 AND vector_status=0')
    print(c.fetchone()[0])
conn.close()
" 2>/dev/null)
  echo "--- 第 $i 批结束 $(date '+%H:%M:%S')，剩余 $remaining 条 ---" >> "$LOG"
  if [ "$remaining" = "0" ]; then
    echo "=== 全部完成 $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
    echo "ALL DONE"
    break
  fi

  sleep 5
done
