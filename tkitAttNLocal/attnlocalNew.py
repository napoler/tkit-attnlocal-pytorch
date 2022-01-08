# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# import pytorch_lightning as pl

class AttNLocalNew(nn.Module):
    """
    自动限制矩阵

    实现斜对角线保留权重，其他的设为-inf


    """

    def __init__(self, maxlen=128, limit=20):
        super(AttNLocal, self).__init__()
        self.limit = limit
        self.maxlen = maxlen
        pass

    def forward(self, x):
        B, L, D = x.size()
        mask = torch.ones_like(x).tril(diagonal=-1) + torch.ones_like(x).triu(diagonal=self.limit)  # 下三角矩阵
        x[mask == 1] = -float("Inf")
        return x

        pass


if __name__ == "__main__":
    print("start test")
    # 输入维度和长度一样的矩阵
    a = torch.randn(5, 10, 16)
    # print("a", a)
    attL = AttNLocal(10, 5)
    out = attL(a)
    print(a)
    print(out.argmax(-1))
    # print()
