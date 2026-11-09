# AutoDL迁移与顺序实验

更新：2026-11-09。当前顺序：**VOC07+12 → SKU-110K → 完整COCO2017**。每次只准备一个数据集，先基线，再候选与消融；完成归档后再处理下一个。

**已被后续指令替换：yang109恢复，实验已回到原服务器，先收尾VisDrone，再VOC→SKU→COCO。本文保留迁移操作历史，不再是当前启动入口。以下“当前实际状态”指迁移时的快照，不表示AutoDL安装/下载已完成；请勿据此重复启动任务。** 当前操作见[实验进展与运行指南](实验进展与运行指南.md)。

## 当前实际状态

- 已用密码登录AutoDL，验证回环反向代理能访问GitHub；没有修改SSH免密认证设置。
- 代码基于`81b2d0f03ec1a26f98b98903daf7d79fce455469`，工作目录`/root/autodl-tmp/YOLO-Master`，迁移分支`research/autodl-migration-20261109`。
- 独立`yolomaster`环境正在后台安装。复用既有Miniconda，不覆盖base，不修改GPU驱动。
- 完整VOC准备已在后台启动，PID保存在`/root/autodl-tmp/logs/prepare-voc.pid`。使用已有base Python3.12.3/Pillow11.3执行数据整理，训练环境另用Python3.10.20。
- 当前盘仍50GiB；可以先下载VOC原图，尚不能构建约130GiB的全量训练/测试特征缓存。请在AutoDL控制台完成数据盘扩容后用`df -h`确认；只挂载原图不会自动扩容。
- 没有启动新数据集训练、GPU冒烟或子集实验，没有新增AP。旧服务器不可达，未能复制只保存在旧服务器的检查点和缓存；Git只恢复代码与已提交的轻量记录。

## Windows：维持Clash反向转发

Clash本地端口7897。当前Codex已建立远端17897转发，同一端口只保留一个监听。当前会话结束、转发失效后，在Windows PowerShell执行下面一条并保持窗口运行；密码由SSH交互输入。

```powershell
ssh -p 37540 -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:17897:127.0.0.1:7897 root@connect.weste.seetacloud.com
```

若提示远端转发失败，先确认现有隧道是否仍正常，不要重复启动多个相同端口的转发。下载依赖本地Clash、电脑网络及该SSH隧道；关闭窗口或电脑休眠会使代理中断。

## AutoDL终端：查看安装与全量下载

以下命令可直接在VS Code的AutoDL终端运行，不会重复启动任务。

```bash
cat /root/autodl-tmp/transfer/setup-status.txt
tail -n 30 /root/autodl-tmp/transfer/setup.log
ps -p "$(cat /root/autodl-tmp/transfer/setup.pid)" -o pid,etime,args

ps -p "$(cat /root/autodl-tmp/logs/prepare-voc.pid)" -o pid,etime,args
tail -n 8 /root/autodl-tmp/logs/prepare-voc.log
du -sh /root/autodl-tmp/datasets/VOC/archives
df -h / /root/autodl-tmp
```

环境完成标记为`SUCCEEDED environment_and_code`。全量数据完成标记为`/root/autodl-tmp/datasets/VOC/complete.json`，只有所有指定图像解码、标注和数量验证后才写出；压缩包下载完成不等于数据准备完成。

```bash
if [[ -f /root/autodl-tmp/datasets/VOC/complete.json ]]; then
    cat /root/autodl-tmp/datasets/VOC/complete.json
else
    echo "VOC完整数据准备尚未完成；查看日志和进程。"
fi
```

环境成功后进入项目：

```bash
source /root/autodl-tmp/transfer/env.sh
cd /root/autodl-tmp/YOLO-Master
python -V
cat /root/autodl-tmp/transfer/environment.json
git branch --show-current
```

## 中断后恢复准备

仅在原准备进程确实退出后运行。脚本有互斥锁，断点ZIP和已验证文件会复用；存在不一致文件会报错并保留证据，不静默覆盖。

```bash
mkdir -p /root/autodl-tmp/logs
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export NO_PROXY=localhost,127.0.0.1,::1
nohup /root/miniconda3/bin/python -u \
    /root/autodl-tmp/YOLO-Master/online_experiments/prepare_autodl_voc.py \
    --root /root/autodl-tmp/datasets/VOC \
    >> /root/autodl-tmp/logs/prepare-voc.log 2>&1 < /dev/null &
printf '%s\n' "$!" > /root/autodl-tmp/logs/prepare-voc.pid
```

环境安装失败时先看`setup-status.txt`和日志。确认原安装进程退出、代理恢复后才执行以下恢复；不同时运行两次pip/conda安装。

```bash
nohup bash /root/autodl-tmp/transfer/setup_autodl_cvpr.sh \
    >> /root/autodl-tmp/transfer/setup-launch.log 2>&1 < /dev/null &
printf '%s\n' "$!" > /root/autodl-tmp/transfer/setup.pid
```

## 数据规模与评估接入

VOC准备器读取官方`ImageSets`，采用2007+2012 trainval共16551张训练，2007 test共4952张测试、20类。不把2012压缩包额外图像混入检测训练。原始JPEG/XML保留；训练YOLO标签忽略`difficult`，主评估仍读取保留的原XML/评估JSON并遵守difficult规则。坐标转换明确记录为`xmin-1,ymin-1,xmax,ymax`，不裁剪或覆盖原标注。

主端点为预先固定最终轮的VOC2007 AP50（11点插值），额外COCO-style AP另列。测试集不用于每轮选best、调参或早停。准备器只准备数据，没有实现训练与AP评分。

下一步在足够磁盘容量上构建VOC缓存，接入20类头、VOC评估与统一成本记录，再启动完整划分的基线队列。旧`run_paper_baselines.sh`固定VisDrone数量/类别/评估，不能直接把路径改成VOC来运行。新训练轮次、增强、预算与种子要在启动前冻结；不做新的小规模验证。

```text
/root/autodl-tmp/
├── YOLO-Master/                     用户源码和轻量研究记录
├── transfer/
│   ├── setup_autodl_cvpr.sh          环境安装入口
│   ├── setup-status.txt / setup.log
│   ├── setup.pid / env.sh            env.sh在安装成功后生成
│   ├── environment.json / pip-freeze.txt
│   └── source/prepare_autodl_voc.py   本次数据准备源快照
├── logs/prepare-voc.log / prepare-voc.pid
├── datasets/VOC/
│   ├── archives/                    原ZIP及未完成.partial
│   ├── raw/VOCdevkit/               原JPEG、XML、ImageSets
│   ├── images/ / labels/            图片优先硬链，避免双份占盘
│   ├── train.txt / test.txt / voc0712.yaml
│   ├── evaluation/test2007.json     保留所有框及difficult
│   ├── dataset-manifest.json        全量身份、哈希和坐标约定
│   └── complete.json                完整核验后才出现
└── cache/huggingface/                固定教师权重，不含旧检查点
```

## 文件迁移命令

本机没有完整数据/权重，不需要把整个CVPR文件夹上传；聊天记录、PDF、凭据也不打包。代码已经从自己的Git分支恢复。需要补传明确文件时，在Windows PowerShell使用交互式密码SCP：

```powershell
scp -P 37540 "C:\Users\jiazonghan\Desktop\CVPR\online_experiments\prepare_autodl_voc.py" root@connect.weste.seetacloud.com:/root/autodl-tmp/transfer/
```

旧服务器网络恢复后，才可在**旧服务器终端**按明确目录推送。以下只展示一个已有数据目录的传输，不带`--delete`，不会替换AutoDL的新VOC目录：

```bash
rsync -aP -e 'ssh -p 37540' \
    /mnt/data/users/zhjia/YOLO-Master/datasets/VisDrone/ \
    root@connect.weste.seetacloud.com:/root/autodl-tmp/datasets/VisDrone/
```

当前不执行旧VisDrone拷贝，按最新安排先做新VOC；也不搬135GiB旧缓存。旧检查点后续需选择确切文件迁移，保留优化器/RNG和原身份；不能从Git的summary恢复训练。AutoDL与旧服务器的GPU、运行环境不同，时间记录分开，不能直接混算受控加速。
