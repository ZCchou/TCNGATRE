# TranAD Bundle

默认共享数据集支持 `..\dataset\alfa` 和 `..\dataset\simulate`，默认训练输出会按数据集写到：

- `runs\tranad_alfa`
- `runs\tranad_simulate`

当前默认 `window_size=32`。

常用命令：

```powershell
python train_tranad.py --dataset alfa
python infer_tranad.py --dataset simulate
python eval_tranad.py --dataset simulate
```

如果不想用默认输出目录，先设置：

```powershell
$env:UAV_TRANAD_RUN_ROOT = ".\\runs\\tranad_new"
```
