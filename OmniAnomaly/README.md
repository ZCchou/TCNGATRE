# OmniAnomaly Bundle

默认共享数据集支持 `..\dataset\alfa` 和 `..\dataset\simulate`，默认训练输出会按数据集写到：

- `runs\omni_anomaly_alfa`
- `runs\omni_anomaly_simulate`

当前默认 `window_size=32`。

这个迁移包里保留了独立的 `eval_omni_anomaly.py`，不再依赖旧的总分析脚本。

常用命令：

```powershell
python train_omni_anomaly.py --dataset alfa
python infer_omni_anomaly.py --dataset simulate
python eval_omni_anomaly.py --dataset simulate
```

如果不想用默认输出目录，先设置：

```powershell
$env:UAV_OA_RUN_ROOT = ".\\runs\\omni_new"
```
