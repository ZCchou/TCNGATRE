import torch
import torch.nn as nn

class TimeMLPEncoding(nn.Module):
    """
    对 Δt 做可学习编码（MLP）。
    输入:
      - (N,)
      - (N, E)
      - (N, E, 1)
    输出:
      - (N, d_model) or (N, E, d_model)
    """
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # ====== [修复] 保证最后一维是 1，才能喂给 Linear(1, hidden) ======
        # dt: (BS, E) -> (BS, E, 1)
        # dt: (E,)    -> (E, 1)
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        elif dt.dim() == 2:
            dt = dt.unsqueeze(-1)
        else:
            # 如果已经是 (..,1) 就不动；否则强制 reshape 成 (..,1)
            if dt.size(-1) != 1:
                dt = dt.unsqueeze(-1)

        dt = dt.to(dtype=torch.float32)
        return self.net(dt)