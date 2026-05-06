# Recurrent AE Bundle

默认共享数据集支持 `..\dataset\alfa` 和 `..\dataset\simulate`，默认训练输出会按数据集写到：

- `runs\recurrent_ae_alfa`
- `runs\recurrent_ae_simulate`

当前默认 `window_size=32`。

常用命令：

```powershell
python train_recurrent_ae.py --dataset alfa
python infer_recurrent_ae.py --dataset simulate
python eval_recurrent_ae.py --dataset simulate
```

如果不想用默认输出目录，先设置：

```powershell
$env:UAV_RAE_RUN_ROOT = ".\\runs\\recurrent_ae_new"
```
