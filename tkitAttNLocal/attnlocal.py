# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# import pytorch_lightning as pl

class AttNLocal(nn.Module):
    """
    自动限制矩阵

    实现斜对角线保留

    """

    def __init__(self, maxlen=128, limit=20):
        super(AttNLocal, self).__init__()
        self.limit = limit
        self.maxlen = maxlen
        pass

    def autoBulidlMaskLimit(self):
        """
        构建一个矩阵
        自动限制 预测长度


        """
        datas = []
        for it in range(self.maxlen):
            new = it * [0] + [1] * self.limit + [0] * self.maxlen
            datas.append(new[:self.maxlen + self.limit])
        return datas

    def forward(self, x):
        B, L, D = x.size()
        m = self.autoBulidlMaskLimit()
        mask = torch.Tensor([m] * B)
        # print(mask.size())
        # torch.where(mask>0,x,mask)
        # 构建填充
        pad = torch.zeros(B, L, self.limit)
        xplus = torch.cat((x, pad.to(x.device)), dim=-1)
        active_loss = mask.view(-1) == 1
        # print(xplus.size())
        xplus_out = xplus.view(-1)[active_loss].view(B, L, -1)
        return xplus_out

        pass


if __name__ == "__main__":
    print("start test")
    # 输入维度和长度一样的矩阵
    a = torch.randn(32, 10, 10)
    print("a", a)
    attL = AttNLocal(10, 5)
    print(attL(a).size())
