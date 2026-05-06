# USAD Bundle

默认共享数据集支持 `..\dataset\alfa` 和 `..\dataset\simulate`，默认训练输出会按数据集写到：

- `runs\usad_alfa`
- `runs\usad_simulate`

这一版 `USAD` 已经改成宽表滑窗输入，不再依赖 `patch_index*.csv`、`npz/` 或 `HistoryFutureSensorDataset` 作为运行时输入。
当前默认窗口设置是 `history_steps=128`、`future_steps=32`，总窗口 `160`。

常用命令：

```powershell
python train_usad.py --dataset alfa
python infer_usad.py --dataset simulate
```

如果不想用默认输出目录，先设置：

```powershell
$env:UAV_USAD_RUN_ROOT = ".\\runs\\usad_new"
```
