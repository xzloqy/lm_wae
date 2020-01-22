#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-10-21 下午9:58
# @Author  : Xinxin Zhang

import  torch
import  torch.nn as nn
loss = nn.MSELoss()
for i in range(100):
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input, target)
    print(output.item())
    output.backward()
