实现方案（待探针结果确认后应用）

## 目标
把 BDHBlockFLA 每层为反向保存的 3 张 [B,T,nh,N] 激活（x_sparse / y_sparse / xy_sparse）
从"必须常驻"改为"反向时重算"，从而把 93% 的显存大头压下来。

## 手段
torch.utils.checkpoint（梯度检查点）：
  - 数学上【精确等价】——反向时用输入重算前向，不是近似
  - 代价：前向算两次（约 +33% 时间）
  - 收益：中间激活不再常驻

## 实现（改 dynfw/models/fused_fw_gdn_fla.py 的 BDHBlockFLA）

1) __init__ 增开关：
       self.grad_ckpt = grad_ckpt      # 默认 False，保持现有行为不变

2) forward 拆成「外壳 + 实现体」：
       def forward(self, x, memories=None):
           if self.grad_ckpt and self.training:
               from torch.utils.checkpoint import checkpoint
               return checkpoint(self._forward_impl, x, memories, use_reentrant=False)
           return self._forward_impl(x, memories)

       def _forward_impl(self, x, memories=None):
           ... 原 forward 主体（循环体不变）...
           return x, memories

3) 构造函数链上把 grad_ckpt 透传：
       BDHBlockFLA(..., grad_ckpt=False)
       BDHBlockFLALM(..., grad_ckpt=False)
       distill_qwen.py 增 --grad-ckpt 开关

## 判据（跑前锁死，事后不改）
  J1 数值等价：开启后 logits / loss 与关闭时一致（checkpoint 是精确重算，应逐位一致；
               若因 bf16 FLA 段引入非确定性，容差按 bf16 量级并如实说明）
  J2 显存：peak 应显著下降（目标：从 5.13GiB 压向 TF 的 0.95GiB 量级）
  J3 速度：允许变慢（重算代价），但须报出确切倍数，不接受"差不多"
  J4 能力：短冒烟 loss 不变；长 T 需重跑确认 5738.0 不漂
  J5 兼容：与 torch.compile 共存（探针验证）
  J6 默认值：grad_ckpt 默认 False ⇒ 现有交付形态与已有读数【不作废】

## 不做的事
  - 不改任何数学（门控 / 记忆更新 / 读出方式一律不动）
  - 不改 FLA 段（它本来就是 chunked，已经省了）
  - 不碰默认行为（避免作废现有台账读数）
