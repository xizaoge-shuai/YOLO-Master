# 校园两路 + AutoDL两路：2026-11-14接手

## 2026-11-16 凌晨更新：中继与断点恢复

本段覆盖下方历史状态。校园VOC INT8已在12 GiB显存准入下开始训练，最近核查6/100轮；SKU Adapter77/100。每次启动还需主机可用内存12 GiB、本项目不超过两路。核心最终结果仍VOC3/7、SKU5/7，指标8/14、失败0；没有因此改变检测器、数据、增强或100轮配方。

端口用途如下，均只绑定回环地址：

| 用途 | 路径 | 依赖 |
|---|---|---|
| AutoDL访问HTTP/HTTPS | AutoDL `127.0.0.1:17897` → Windows Clash `127.0.0.1:7897` | 本机Clash及AutoDL隧道 |
| 校园上传与结果回传 | 校园 `127.0.0.1:19822` → Windows `127.0.0.1:19022` → AutoDL SSH 22 | 本机、aTrust及两段SSH中继 |

原生SSH隧道曾在11月15日23:16断开。现已改为独立后台中继PID25632，现场CONNECTED、AutoDL代理HTTP200、校园19822到AutoDL SSH验证通过；新上传3439657/回传3439658已启动。完成tar被复用，SKU缓存从原5,665,357,824字节断点校验后继续增长，TRANSFERRING；AutoDL两项仍WAITING_TRANSFER。真实8 MiB探测约1.78 MiB/s不能当作全批速度；最新大文件窗口证据为research/checks/relay-admission-20261115/relay-sustained-window.json。

### Windows本地：启动或查看中继

在本机PowerShell执行。脚本使用工作区隔离依赖，不改训练服务器环境；密码仅在终端交互输入，不放入命令、环境变量或文件。`launch`把中继放入后台，重复启动会被已有监听端口拒绝。

```powershell
Set-Location -LiteralPath 'C:\Users\jiazonghan\Desktop\CVPR'
python online_experiments/windows_autodl_relay.py status

# 仅在未运行时执行；按提示交互输入AutoDL密码。
python online_experiments/windows_autodl_relay.py launch
python online_experiments/windows_autodl_relay.py status

Get-Content -LiteralPath 'research\checks\relay-admission-20261115\windows-relay-status.json'
```

中继守护程序负责保持/重连隧道。复制和结果回传期间Windows与aTrust仍须在线；退出本聊天不等于可以关机。数据完整到齐后，AutoDL训练本体独立运行；训练结束后回传仍需恢复此链路。

### 校园服务器：查看或恢复传输客户端

先看状态和进程；只有确认原上传/回传客户端均已退出、中继可用后才执行`launch`。不删除partial，不手动改manifest，不重复启动训练。`launch`交互认证后恢复原SHA校验及断点上传；前缀哈希校验阶段offset短暂不变属于正常现象。

```bash
cd /mnt/data/users/zhjia/YOLO-Master-paper-baselines-20261107
PY=/mnt/data/users/zhjia/miniconda3/envs/yolomaster/bin/python
J=runs/paper/migration-autodl/job-20261114-two-lanes

cat "$J/launch.json"
cat "$J/transfer-state.json"
cat "$J/return-status.json"
pgrep -af 'run_autodl_upload.py|return_autodl_results.py'

# 只在旧客户端已退出、中继已可用时运行；密码由终端getpass读取。
PYTHONPATH=.research-deps/autodl-transfer/site:. "$PY" online_experiments/autodl_relay_client.py launch
```

隧道重连与传输恢复是两件事：已FAILED的客户端不会因隧道重连自动复活。查看VOC/SKU校园训练和AutoDL真实续训的命令见[最新进展](实验进展_20261115.md)。本次CPU测试只验证调度、断点和中继代码，没有另做GPU冒烟训练或小数据验证。

在AutoDL确有联网下载需求时，可在对应终端设置代理；既有离线训练不需要为此重启：

```bash
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
```

## 以下为11月14日至15日历史交接记录

2026-11-15更新：完整迁移包已封存；两个源暂停进程已退役，资源已释放。后台传输、AutoDL自动等待/续训及回传客户端均已实际启动，当前为WAITING_TRANSFER；校园VOC FroFA和SKU INT8已完成，SKU继续Adapter，VOC接续其余方法。最新实时命令与解释见[今日进展](实验进展_20261115.md)，以下原暂停现场描述保留为历史。

用户最新要求是停止校园额外两路，并在AutoDL续跑它们。校园保留FroFA/INT8；迁移VOC LOFF-TA、SKU Detail25，固定seed0、100轮、batch8、640和完整原划分。

当前已确认：两迁移进程处于SIGSTOP，完整检查点为LOFF9轮/Detail17轮；原FroFA/INT8继续。AutoDL连接成功且缺失依赖已安装，GPU/环境导入验证、数据复制与新训练分别记录。不能把等待/复制状态当成训练已经开始。

## 校园服务器

项目`/mnt/data/users/zhjia/YOLO-Master-paper-baselines-20261107`。

```bash
cd ~/YOLO-Master-paper-baselines-20261107
J=runs/paper/migration-autodl/job-20261114-two-lanes
cat "$J/paused.json"
test ! -f "$J/transfer-state.json" || cat "$J/transfer-state.json"
test ! -f "$J/return-status.json" || cat "$J/return-status.json"
tail -n 20 "$J/prepare-bundle.log"
```

校园原训练日志仍在`runs/paper/priority/job-20261112-single-seed/{voc0712,sku110k}/train-<method>.log`。新接管器将明确跳过迁移项，避免再次启动。源暂停现场退役前须完成备份/清单封存，退役记录为`$J/retirement.json`。不要手动恢复旧四路监督器。

## AutoDL

SSH入口仍为`ssh -p 37540 root@connect.weste.seetacloud.com`。密码不写入文档、脚本或argv。

新工作盘根目录`/root/autodl-tmp/campus-mirror`；旧绝对路径`/mnt/data/users/zhjia`只通过新建符号链接映射到此目录，不覆盖已有数据。新运行目录如下：

```bash
M=/root/autodl-tmp/campus-mirror
J="$M/.migration/job-20261114-resume"
cat "$J/sku110k/status.json"
cat "$J/voc0712/status.json"
tail -n 20 "$J/sku110k/resume.log"
tail -n 20 "$J/voc0712/resume.log"
nvidia-smi
```

状态顺序是WAITING_TRANSFER→UNPACKING→VERIFYING→TRAINING→SUCCEEDED，失败保留FAILED及日志，不自动重做训练。SKU数据先到即可独立续训，无需等VOC传完。目标容器实际62GiB内存/16CPU、GPU32GiB、数据盘450GiB；宿主`free`显示的512GiB不是容器额度。

## 一次预测收齐指标

每项原训练器只保存final100完整测试预测。目标成功后回传器逐SHA获取结果，保留校园原暂停输出，再发布完整canonical端点；原CPU队列从这些预测计算mAP/AP50/AP75、尺度/类别、AR/PR及已记录资源，不重复推理。

```bash
cd ~/YOLO-Master-paper-baselines-20261107
PY=/mnt/data/users/zhjia/miniconda3/envs/yolomaster/bin/python
"$PY" online_experiments/metric_bundle_campaign.py status \
  --job runs/paper/metric-bundles/job-20261112-single-pass
```

迁移保留原checkpoint、优化器/调度器、RNG、schedules和全部teacher查询日志。未完整保存的半轮需从上个完整轮重放，实际查询成本不能抹去。原环境移存environment-origin.json，新environment.json记录真实AutoDL环境。跨硬件不保证逐位一致，累计墙钟时间是混合硬件观测值，不能包装成受控提速。
