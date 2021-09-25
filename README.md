# tkitAttNLocal

构建对角线mask矩阵，用来降维相对位置数据。


安装
> pip install tkitAttNLocal


```python

import torch.nn as nn
import torch
import torch.nn.functional as F
from tkitAttNLocal import AttNLocal

# 输入维度和长度一样的矩阵
a=torch.randn(32,100,100)

attL=AttNLocal(100,10)
attL(a).size()

```

> torch.Size([32, 100, 10])