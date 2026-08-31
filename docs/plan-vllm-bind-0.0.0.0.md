# 实施方案：vLLM (qwen38 gptq mtp3) 监听地址改为 0.0.0.0

> 交给 opencode 执行。目标：让 vLLM OpenAI API 从本机和 Tailscale 内网都能访问。

## 1. 目标

当前实例：模型 `qwen38-gptq-fp16kv-128K-mtp3-text-only`，端口 8000，
由独立脚本 `scripts/start_qwen38_gptq_fp16kv_mtp3.sh` 直接启动（**未**经过 launcher.sh 管理，run-logs 下没有它的 .pid 文件）。

改完后必须同时满足：
1. `ss -tln | grep :8000` 显示监听地址为 `0.0.0.0:8000`（不再是 127.0.0.1）
2. 本机仍然可用：`curl http://127.0.0.1:8000/v1/models` 返回正常（pi 的 `~/.pi/agent/models.json` 里 baseUrl 就是 127.0.0.1:8000，**不要改它**；0.0.0.0 绑定天然覆盖回环地址）
3. Tailscale 可达：`curl http://100.64.0.4:8000/v1/models` 返回 200 JSON（改动前该地址是 connection refused）

## 2. 代码改动（唯一需要改的文件）

**文件：`scripts/start_qwen38_gptq_fp16kv_mtp3.sh`**

(a) 在文件头部 `LOG_DIR=...` 一行之后，增加可覆盖的默认值：

```bash
HOST="${HOST:-0.0.0.0}"          # 监听地址，可用环境变量覆盖
VLLM_API_KEY="${VLLM_API_KEY:-}" # 可选：非空时启用 API key 鉴权
```

(b) 把启动参数里的：

```bash
  --host 127.0.0.1 \
```

改成：

```bash
  --host "${HOST}" \
```

(c) 可选加固（推荐）：如果 `VLLM_API_KEY` 非空，向 vLLM 追加 `--api-key "$VLLM_API_KEY"`（用 VLLM_ARGS 数组方式条件追加，或直接在参数行前展开一个数组）。
⚠️ 启用 key 后所有客户端都要带 `Authorization: Bearer <key>`，同时需把 `~/.pi/agent/models.json` 中 openai provider 的 `"apiKey": "not-needed"` 改成同一个 key，否则 pi 本地调用会 401。
**默认按"不启用 key"执行**，除非执行者明确要求开 key。

## 3. 重启步骤（严格按顺序）

重启 = 杀掉旧进程组后重新运行改好的脚本。所有旧进程（bash 脚本、api_server、EngineCore、Worker_TP0/TP1、tee）共享同一个 PGID（旧 bash 是进程组组长）。

```bash
cd /home/lingyuzeng/project/vllm-2080ti

# 1) 找到顶层 bash 脚本 pid 及其 PGID
OLD_PID=$(pgrep -f 'bash .*start_qwen38_gptq_fp16kv_mtp3.sh' | head -1)
PGID=$(ps -o pgid= -p "$OLD_PID" | tr -d ' ')

# 2) 对整个进程组发 TERM，等最多 60 秒退出
kill -TERM -- -"$PGID"
for i in $(seq 1 60); do
  ps -o pid= -g "$PGID" 2>/dev/null | grep -q . || break; sleep 1
done
# 仍有残留则强杀
ps -o pid= -g "$PGID" 2>/dev/null | grep -q . && kill -9 -- -"$PGID"

# 3) 确认 8000 端口已释放、显存已回收
ss -tln | grep ':8000' && echo "端口未释放，排查" # 应无输出
nvidia-smi  # 该卡显存应归零或接近 0

# 4) 重新启动（脱离终端，新脚本内自带 tee 写 run-logs/vllm-...-<新时间戳>.log）
cd /home/lingyuzeng/project/vllm-2080ti
nohup setsid bash scripts/start_qwen38_gptq_fp16kv_mtp3.sh >/dev/null 2>&1 & disown

# 5) 记录 pid 便于以后管理（沿用项目 pid 文件命名惯例）
sleep 5; NEW_PID=$(pgrep -f 'vllm.entrypoints.openai.api_server' | head -1)
echo "$NEW_PID" > run-logs/vllm-qwen38-gptq-fp16kv-mtp3-text-only.pid

# 6) 轮询等待就绪（27B 模型冷启动，历史经验 3~8 分钟；上限给 10 分钟）
for i in $(seq 1 60); do
  curl -s --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "READY after ~$((i*10))s"; break; }
  sleep 10
done
```

注意：若届时正在跑的 pi 会话（本终端所属的 agent）依赖这个模型，重启期间它不可用，属预期现象。

## 4. 验收标准（全部满足才算完成）

```bash
# 1) 监听地址
ss -tln | grep ':8000'            # 期望: 0.0.0.0:8000（或 [::]:8000）

# 2) 本机
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool | head

# 3) Tailscale 内网（改动前是 connection refused / exit 7）
curl -s --max-time 5 http://100.64.0.4:8000/v1/models

# 4) 端到端对话
curl -s http://100.64.0.4:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-gptq-fp16kv-128K-mtp3-text-only","messages":[{"role":"user","content":"ping"}],"max_tokens":10}'
# 期望: 200，choices[0] 有返回内容

# 5) 新日志文件存在且无 ERROR（run-logs/ 下最新的 vllm-qwen38-*.log）
```

有条件的话，再从 tailnet 上的**另一台设备**跑一次上面的 curl 做跨机复核。

## 5. 回滚方案

改動通过 `HOST` 环境变量驱动，回滚不需要再改文件：

```bash
# 若 0.0.0.0 有任何问题，用同样方式重启并覆盖回本机监听
HOST=127.0.0.1 nohup setsid bash scripts/start_qwen38_gptq_fp16kv_mtp3.sh >/dev/null 2>&1 & disown
```

## 6. 必须知晓的事（别踩坑）

1. **安全面**：0.0.0.0 = 所有接口都开。本机物理网卡有 `192.168.8.231` 和 `192.168.9.52` 两个 LAN 地址，改完后**同一局域网任何设备**也能直接调用这个 8 卡 27B 推理服务。vLLM OpenAI server 默认**无鉴权**（/v1/models、/v1/chat/completions 全裸奔）。如局域网不可信，优先采用：
   - 方案 A（加入 step 2c 的 api-key），或
   - 方案 B（默认 HOST 改成 `100.64.0.4`，只绑 Tailscale 接口）——注意 B 的副作用：tailscale 断开时服务启动会直接 bind 失败，会拖累本地使用。
   本方案按用户要求默认 0.0.0.0，执行者如被要求加固则走 A。
2. **IPv6**：`0.0.0.0` 只绑 IPv4。Tailscale 还有 `fd7a:...` 的 IPv6 地址，经 IPv6 访问 8000 会拒绝。如需双栈，把默认 HOST 设为 `::`（uvicorn 会双栈监听），其余验收命令不变。默认不动。
3. **不要动** `~/.pi/agent/models.json` 的 baseUrl（保持 127.0.0.1:8000）。
4. **不要**把 instance 迁去 launcher.sh 管理（SERVICE_SCOPE=lan 机制存在，但 qwen38 路线没有 profile，迁移超出本次范围）。
5. 修改脚本前先备份：`cp scripts/start_qwen38_gptq_fp16kv_mtp3.sh{,.bk-$(date +%Y%m%d)}`
