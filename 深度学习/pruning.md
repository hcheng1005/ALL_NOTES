# 模型剪枝的一些尝试

- [什么是模型剪枝](#什么是模型剪枝)
- [Torch-Pruning](#torch-pruning)
- [pointpillars剪枝实践](#pointpillars剪枝实践)
  - [模型训练](#模型训练)
  - [Backbone重定义](#backbone重定义)
  - [模型剪枝](#模型剪枝)
  - [模型微调](#模型微调)
- [后续工作](#后续工作)
- [参考资料](#参考资料)

---

## 什么是模型剪枝

[模型压缩-剪枝算法详解](https://zhuanlan.zhihu.com/p/622519997)

模型剪枝（Pruning）也叫模型稀疏化，不同于模型量化对每一个权重参数进行压缩，稀疏化方法是尝试直接“删除”部分权重参数。模型剪枝的原理是通过剔除模型中 “不重要” 的权重，使得模型减少参数量和计算量，同时尽量保证模型的精度不受影响。

## Torch-Pruning

[CVPR 2023 | DepGraph 通用结构化剪枝](https://zhuanlan.zhihu.com/p/619146631)

[Torch-Pruning | 轻松实现结构化剪枝算法](https://zhuanlan.zhihu.com/p/619482727)

```
Torch-Pruning（TP）是一个结构化剪枝库，与现有框架（例如torch.nn.utils.prune）最大的区别在于，TP会物理地移除参数，同时自动裁剪其他依赖层。TP是一个纯 PyTorch 项目，实现了内置的计算图的追踪(Tracing)、依赖图(DepenednecyGraph, 见论文)、剪枝器等功能，同时支持 PyTorch 1.x 和 2.0 版本。本章节主要介绍 Torch-Pruning 的一些基础功能。
```


## pointpillars剪枝实践

```
剪枝三步骤：训练、剪枝、微调
```

### 模型训练

首先基于KITTI数据集训练出一版本原始pointpillars模型和权重文件。

### Backbone重定义

PointPillars模型剪枝对象为`BaseBEVBackbone`结构。并且，为了方便使用`torchinfo`中的`summary`进行模型参数的估计，重新定义`BaseBEVBackbone`结构。（就是copy一份，把forward中的输入参数有dict类型改为numpy类型）

```python
def forward(self, spatial_features):
    """
    Args:
        spatial_features
    Returns:
    """

    # spatial_features = data_dict['spatial_features']
    ups = []
    x = spatial_features
    ...
```

定义新的backbone，并输出其模型参数量
```python
# 定义新的BaseBEVBackbone
myModel_BaseBEVBackbone = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, 64)
myModel_BaseBEVBackbone.cuda()
example_inputs = torch.randn(1,64,432,496).cuda()
summary(myModel_BaseBEVBackbone, input_data=example_inputs)
```

### 模型剪枝

**Step 1: 定义重要性评估标准**
```python
# 对BaseBEVBackbone剪枝
# step 1: 定义重要性评估标准
imp = tp.importance.GroupNormImportance()
```


**Step 2: 遍历模型结构，摘除不需要进行剪枝处理的子模块**
```python
# 摘除不进行剪枝的结构
ignored_layers = []
for m in myModel_BaseBEVBackbone.modules():
    if isinstance(m, torch.nn.ConvTranspose2d):
        ignored_layers.append(m)
        
# print("ignored_layers: ", ignored_layers)
```


**Step 3: 定义剪枝器，目标稀疏度以及剪枝迭代次数**
```python
# 定义剪枝器
iterative_steps = 5
pruner = tp.pruner.GroupNormPruner(
    myModel_BaseBEVBackbone,
    example_inputs, # 用于分析依赖的伪输入
    importance=imp, # 重要性评估指标
    iterative_steps=iterative_steps, # 迭代剪枝，设为1则一次性完成剪枝
    pruning_ratio=0.2, # 目标稀疏性
    ignored_layers=ignored_layers)
```


**Step 4: 执行剪枝，观察中间日志**
```python
# 执行剪枝
base_macs, base_nparams = tp.utils.count_ops_and_params(myModel_BaseBEVBackbone, example_inputs)
for i in range(iterative_steps):
    pruner.step() # 执行裁剪，每次会裁剪的比例为 [ch_sparsity / iterative_steps]
    macs, nparams = tp.utils.count_ops_and_params(myModel_BaseBEVBackbone, example_inputs)
    print("  Iter %d/%d, Params: %.2f M => %.2f M" % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
    print("  Iter %d/%d, MACs: %.2f G => %.2f G"% (i+1, iterative_steps, base_macs / 1e9, macs / 1e9))
    print("  Iter %d/%d, Pruning ratio: %.2f " % (i+1, iterative_steps, nparams / base_nparams))

# # 再次查看模型结构以及参数量    
# print(myModel_BaseBEVBackbone)
# summary(myModel_BaseBEVBackbone, input_data=example_inputs)  
```

### 模型微调
经过**模型剪枝**后，得到一个稀疏化的backbone，首先将其替换到原模型中。

```python
# 方式一
# model.module_list[2] = copy.deepcopy(myModel_BaseBEVBackbone)  # TBD 需对齐子模块输入输出

# 方式二: 直接替换内部modules
model.backbone_2d.blocks = copy.deepcopy(myModel_BaseBEVBackbone.blocks)
model.backbone_2d.deblocks = copy.deepcopy(myModel_BaseBEVBackbone.deblocks)
# print(model.backbone_2d) # 剪枝后的模型结构
# print(model)
```


评估模型，此时结果会很差，因此需要微调（重新训练少数几个epoch）。
```python
# 剪枝后评估模型效果
model = model.cuda().eval()
eval_model(model, cfg, args, logger)
    
# 重新训练
re_trainModel(model)

# 再次评估模型效果
eval_model(model, cfg, args, logger)
```


## 后续工作
上述内容只是简单调用了torch_pruning库中的`GroupNormPruner`对模型进行剪枝，流程比较粗暴且黑盒。

后续将继续尝试其他模型剪枝方法和策略，比如`tp.pruner.MagnitudePruner()`
和`tp.pruner.BNScalePruner()`.


## 参考资料
- [CVPR 2023 | DepGraph 通用结构化剪枝](https://zhuanlan.zhihu.com/p/619146631)

- [Torch-Pruning | 轻松实现结构化剪枝算法](https://zhuanlan.zhihu.com/p/619482727)

- [torch-pruning的基本使用](https://blog.csdn.net/magic_ll/article/details/134441473)

- [TAO Toolkit - PyTorch Backend](https://github.com/NVIDIA/tao_pytorch_backend)