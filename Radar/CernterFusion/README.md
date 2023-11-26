# CenterFusion
> 一种利用雷达和摄像机数据进行三维目标检测的中间融合方法。

Paper: [CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection](https://arxiv.org/abs/2011.04841)

Code: [CenterFusion](https://github.com/hcheng1005/CenterFusion.git)

Code: [CenterFusionWithROS](https://github.com/hcheng1005/CenterFusionWithROS)

---

- [概述](#概述)
- [Camera部分](#camera部分)
- [Radar部分](#radar部分)
  - [预处理](#预处理)
  - [**截锥关联**](#截锥关联)
- [Q\&A](#qa)
  - [数据预处理(即训练模型下)中如何构造hm真值](#数据预处理即训练模型下中如何构造hm真值)
    - [self.\_load\_pc\_data](#self_load_pc_data)
    - [self.\_process\_pc](#self_process_pc)
    - [self.\_add\_instance](#self_add_instance)
  - [推理过程如何计算pc\_hm](#推理过程如何计算pc_hm)
- [参考链接](#参考链接)

---
# TODO

- [x] 截锥关联的代码解读与注释
- [ ] C++部署工程实践

---

# 模型结构

![](images/20231113203556.png)

## 概述

算法流程：

1. 首先使用**CenterNet算法**进利用摄像头数据预测目标的中心点，并回归得到**目标的3D坐标、深度、旋转等信息**

2. 然后作者将雷达检测到的目标数据和上面检测到的目标中心点进行关联，使用了**截锥的方法**

3. 将**关联后的目标的特征和雷达数据检测到的深度和速度信息组成的特征图并联**，在进行3D目标深度、旋转、速度和属性的回归

其中关键步骤是CenterNet检测结果与雷达点目标的关联，在三维空间视锥法找到对应目标的示意图：

![](images/20231113204418.png)

# 模型解读

## Camera部分

- CenterNet中主要提供了三个骨干网络 ResNet-18 (ResNet-101)、**DLA-34**、Hourglass-104

- CenterFusion网络架构在对图像进行初步检测时，**采用CenterNet网络中修改版本的骨干网络DLA**（深层聚合）作为全卷积骨干网，来提取图像特征，因为DLA网络大大减少了培训时间，同时提供了合理的性能再回归图像特征来预测图像上的目标中心点，以及目标的2D大小（宽度和高度）、中心偏移量、3D尺寸、深度和旋转
  
- 主要回归头的组成：256个通道的3×3卷积层、1×1卷积层。这为场景中每个被检测到的对象提供了一个精确的2D边界框以及一个初步的3D边界框.

## Radar部分

### 预处理
- 首先，将毫米波雷达点云转换至图像坐标系，并删除像素外的点云；
  - 相关代码位置：generic_dataset.py -> _load_pc_data
  
- 此次，由于原本的雷达点云没有高度信息，因此作者对点云进行**支柱扩张**处理，就是将每个雷达点云构造成一个个体柱pillars（默认参数：1.5, 0.5, 0.5）；
  - 相关代码位置：generic_dataset.py -> _process_pc
  
- 最后，构造**雷达点云heatmap信息**
  - 相关代码位置：generic_dataset.py -> _process_pc

### **截锥关联**

![](images/20231113210446.png)

- **第1步**：利用图像平面中对象的2D边界框及其估计深度和大小，为对象创建一个3D感兴趣区域（RoI）截锥。
  
- **第2步**：对于每一个与物体相关的雷达检测，我们生成三个以物体的二维包围框为中心并在其内部的热图通道，热图的宽度和高度与物体的 2D 边界框成比例，并由参数α控制其中热图值是归一化对象深度d，也是自中心坐标系中径向速度vx和vy的x和y分量。
  
- **第3步**：两个物体的热图区域重叠，深度值较小的那个占优势，因为只有最近的物体在图像中是完全可见的，如下图所示
  
  ![](images/20231113210816.png)

- 生成的热图然后连接到图像特征作为额外的通道，这些特征被用作二次回归头的输入，以重新计算对象的深度和旋转，以及速度和属性二次回归头由3个卷积层(3×3核)和1×1卷积层组成，以产生所需的输出。

## Q&A

### 数据预处理(即训练模型下)中如何构造hm真值

首先，梳理下数据集的构造过程。

#### self._load_pc_data

函数`def __getitem__(self, index)`用于生成训练需要的所有数据。

其中
```python
pc_2d, pc_N, pc_dep, pc_3d = self._load_pc_data(img, img_info, 
                                                trans_input, trans_output, flipped)
```
是**通过真值图像来关联毫米波点云并生成相关数据**。

生成的数据如下：

1. pc_2d：点云对应的像素坐标系，格式：[wid,height,depth]
2. pc_N：点云个数
3. pc_dep：点云深度特征（实际上是pc_hm_feat）
4. pc_3d：原始点云信息

该函数已经生成了训练所需的hm信息，进一步分析相关函数：

#### self._process_pc

```python
pc_2d, pc_3d, pc_dep = self._process_pc(pc_2d, pc_3d, img, inp_trans, out_trans, img_info)
```

该函数包含如下步骤：
1. 根据输出的size进一步筛选pc_2d
2. 为pc_3d生成pillars（同样转换到像素坐标系并根据输出size进行筛选）
3. 然后为每一个pillars所在的像素坐标范围（即这个pillar在图像上的2Dbox范围）填充点云深度信息，为后续构造pc_hm做准备。
   1. 如下所示
   ```python
    if feat == 'pc_dep':
          channel = self.opt.pc_feat_channels['pc_dep']
          pc_hm_feat[channel, b[0]:b[1], b[2]:b[3]] = depth # 在对应的位置上赋值 depth
   ```

具体生成hm的方法源代码中有两种：
<details>
  <summary>view code</summary>

```python
if self.opt.pc_roi_method == "pillars": # 基于pillars的方式
    wh = pillar_wh[:,i]  # 获取pillar的长宽 width and height
    b = [max(ct[1]-wh[1], 0), 
        ct[1], 
        max(ct[0]-wh[0]/2, 0), 
        min(ct[0]+wh[0]/2, self.opt.output_w)]
    b = np.round(b).astype(np.int32) # 四舍五入，因为这里需要用的是整形的像素坐标
        
elif self.opt.pc_roi_method == "hm": # 基于hm的方式
    radius = (1.0 / depth) * self.opt.r_a + self.opt.r_b
    radius = gaussian_radius((radius, radius)) # 高斯径向基
    radius = max(0, int(radius))
    x, y = ct_int[0], ct_int[1]
    height, width = pc_hm_feat.shape[1:3]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    b = np.array([y - top, y + bottom, x - left, x + right])
    b = np.round(b).astype(np.int32)
```
</details>

**最后生成的pc_hm_feat包含三个信息：['pc_dep', 'pc_vx', 'pc_vz']**。

#### self._add_instance
看下函数入参：
```python
self._add_instance(
  ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
  calib, pre_cts, track_ids)
```
其中的`ret`就包含之前生成的各类信息，比如`'image'、'pc_2d'、'pc_dep'`等等，通过该函数，将`pc_dep`生成最后的`pc_hm`，具体实现如下：

<details>
  <summary>view code</summary>

```python
if self.opt.pointcloud:
  ## get pointcloud heatmap
  if self.opt.disable_frustum:
    ret['pc_hm'] = ret['pc_dep']
    if opt.normalize_depth:
      ret['pc_hm'][self.opt.pc_feat_channels['pc_dep']] /= opt.max_pc_dist
  else:
    dist_thresh = get_dist_thresh(calib, ct, ann['dim'], ann['alpha'])
    pc_dep_to_hm(ret['pc_hm'], ret['pc_dep'], ann['depth'], bbox, dist_thresh, self.opt)
```
</details>


分析其中`def pc_dep_to_hm(pc_hm, pc_dep, dep, bbox, dist_thresh, opt)`该函数：

<details>
  <summary>view code</summary>

```python
'''
names: pc_dep_to_hm
description: Briefly describe the function of your function
param {*} pc_hm
param {*} pc_dep
param {*} dep
param {*} bbox
param {*} dist_thresh
param {*} opt
return {*}
'''
def pc_dep_to_hm(pc_hm, pc_dep, dep, bbox, dist_thresh, opt):
    if isinstance(dep, list) and len(dep) > 0:
      dep = dep[0]
    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    bbox_int = np.array([np.floor(bbox[0]), 
                         np.floor(bbox[1]), 
                         np.ceil(bbox[2]), 
                         np.ceil(bbox[3])], np.int32)# format: xyxy

    # 根据box的size确定ROI范围
    roi = pc_dep[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
    pc_dep = roi[opt.pc_feat_channels['pc_dep']]
    pc_vx = roi[opt.pc_feat_channels['pc_vx']]
    pc_vz = roi[opt.pc_feat_channels['pc_vz']]

    nonzero_inds = np.nonzero(pc_dep)
    
    if len(nonzero_inds[0]) > 0:
    #   nonzero_pc_dep = np.exp(-pc_dep[nonzero_inds])
      nonzero_pc_dep = pc_dep[nonzero_inds]
      nonzero_pc_vx = pc_vx[nonzero_inds]
      nonzero_pc_vz = pc_vz[nonzero_inds]

      ## Get points within dist threshold
      within_thresh = (nonzero_pc_dep < dep+dist_thresh) \
              & (nonzero_pc_dep > max(0, dep-dist_thresh))
      pc_dep_match = nonzero_pc_dep[within_thresh]
      pc_vx_match = nonzero_pc_vx[within_thresh]
      pc_vz_match = nonzero_pc_vz[within_thresh]

      if len(pc_dep_match) > 0:
        arg_min = np.argmin(pc_dep_match) # 寻找最近点
        dist = pc_dep_match[arg_min]
        vx = pc_vx_match[arg_min]
        vz = pc_vz_match[arg_min]
        if opt.normalize_depth:
          dist /= opt.max_pc_dist

        w = bbox[2] - bbox[0]
        w_interval = opt.hm_to_box_ratio*(w)
        w_min = int(ct[0] - w_interval/2.)
        w_max = int(ct[0] + w_interval/2.)
        
        h = bbox[3] - bbox[1]
        h_interval = opt.hm_to_box_ratio*(h)
        h_min = int(ct[1] - h_interval/2.)
        h_max = int(ct[1] + h_interval/2.)

        # 获取对应的属性并赋值到pc_hm中
        pc_hm[opt.pc_feat_channels['pc_dep'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = dist
        pc_hm[opt.pc_feat_channels['pc_vx'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vx
        pc_hm[opt.pc_feat_channels['pc_vz'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vz
```
</details>

大致流程如下：
1. 根据box生成像素范围
2. 根据像素范围筛选pc_feat
3. 找到范围内最近pc_dep最近的索引
4. 获取对应的'pc_dep'、'pc_vx'、'pc_vz'生成pc_hm特征


### 推理过程如何计算pc_hm

首先看下模型推理过程：
```python
# 模型推理
pre_hms, pre_inds = None, None
...
...
output, dets, forward_time = self.process(
        images, self.pre_images, pre_hms, pre_inds, return_time=True, pc_dep=pc_dep, meta=meta)
        ...
        ...


def process(self, images, pre_images=None, pre_hms=None,
    pre_inds=None, return_time=False, pc_dep=None, meta=None):
    with torch.no_grad():
      calib = torch.from_numpy(meta['calib']).float().to(images.device).squeeze(0)
      torch.cuda.synchronize()
      # 推理
      output = self.model(images, pc_dep=pc_dep, calib=calib)[-1]
      ...
      ...


def forward(self, x, pc_hm=None, pc_dep=None, calib=None)
```

从上述流程可以看出，**推理时的入参仅有pc_dep，而不包含pre_hms。**
> [关于`pc_dep`的生成方式前文已经描述过。](#self_load_pc_data)

推理过程中的pc_hm需在后续流程中创建。

在`forward(self, x, pc_hm=None, pc_dep=None, calib=None)`函数中，有如下过程：

```python
# 引入毫米波点云head，获取对应的结果
if self.opt.pointcloud: # 雷达点云存在时生成radar heatmap和second head
  ## get pointcloud heatmap
  # 推理模式下，首先需生成hm 
  # 训练模式下，已经提前预处理好[trainer.py: LINE: 124]）
  if not self.training:
    if self.opt.disable_frustum:
      pc_hm = pc_dep
      if self.opt.normalize_depth:
        pc_hm[self.opt.pc_feat_channels['pc_dep']] /= self.opt.max_pc_dist
    else:
      # 截锥关联并生成hm
      pc_hm = generate_pc_hm(z, pc_dep, calib, self.opt)
      
  ind = self.opt.pc_feat_channels['pc_dep']
  z['pc_hm'] = pc_hm[:,ind,:,:].unsqueeze(1)

  ## Run the second stage heads  
  ## 二阶段检测头【数据特征加上了毫米波点云】
  sec_feats = [feats[s], pc_hm]
  sec_feats = torch.cat(sec_feats, 1)
  for head in self.secondary_heads: 
    z[head] = self.__getattr__(head)(sec_feats)
```
其中，`generate_pc_hm(z, pc_dep, calib, self.opt)`就是生成pc_hm的函数。

进一步解析：

<details>
  <summary>view code</summary>

```python
'''
names: generate_pc_hm
description: 截锥关联并生成heatmap
param {*} output
param {*} pc_dep
param {*} calib
param {*} opt
return {*}
'''
def generate_pc_hm(output, pc_dep, calib, opt):
      K = opt.K
      # K = 100
      heat = output['hm']
      wh = output['wh']
      pc_hm = torch.zeros_like(pc_dep)

      batch, cat, height, width = heat.size()
      scores, inds, clses, ys0, xs0 = _topk(heat, K=K)
      xs = xs0.view(batch, K, 1) + 0.5
      ys = ys0.view(batch, K, 1) + 0.5
      
      ## Initialize pc_feats
      pc_feats = torch.zeros((batch, len(opt.pc_feat_lvl), height, width), device=heat.device)
      dep_ind = opt.pc_feat_channels['pc_dep']
      vx_ind = opt.pc_feat_channels['pc_vx']
      vz_ind = opt.pc_feat_channels['pc_vz']
      to_log = opt.sigmoid_dep_sec
      
      ## get estimated depths
      out_dep = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      dep = _tranpose_and_gather_feat(out_dep, inds) # B x K x (C)
      if dep.size(2) == cat:
        cats = clses.view(batch, K, 1, 1)
        dep = dep.view(batch, K, -1, 1) # B x K x C x 1
        dep = dep.gather(2, cats.long()).squeeze(2) # B x K x 1

      ## get top bounding boxes
      wh = _tranpose_and_gather_feat(wh, inds) # B x K x 2
      wh = wh.view(batch, K, 2)
      wh[wh < 0] = 0
      if wh.size(2) == 2 * cat: # cat spec
        wh = wh.view(batch, K, -1, 2)
        cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
        wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2
      bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                          ys - wh[..., 1:2] / 2,
                          xs + wh[..., 0:1] / 2, 
                          ys + wh[..., 1:2] / 2], dim=2)  # B x K x 4
      
      ## get dimensions
      dims = _tranpose_and_gather_feat(output['dim'], inds).view(batch, K, -1)

      ## get rotation
      rot = _tranpose_and_gather_feat(output['rot'], inds).view(batch, K, -1)

      ## Calculate values for the new pc_hm
      clses = clses.cpu().numpy()

      for i, [pc_dep_b, bboxes_b, depth_b, dim_b, rot_b] in enumerate(zip(pc_dep, bboxes, dep, dims, rot)):
        alpha_b = get_alpha(rot_b).unsqueeze(1)

        if opt.sort_det_by_dist:
          idx = torch.argsort(depth_b[:,0])
          bboxes_b = bboxes_b[idx,:]
          depth_b = depth_b[idx,:]
          dim_b = dim_b[idx,:]
          rot_b = rot_b[idx,:]
          alpha_b = alpha_b[idx,:]

        for j, [bbox, depth, dim, alpha] in enumerate(zip(bboxes_b, depth_b, dim_b, alpha_b)):
          clss = clses[i,j].tolist()
          ct = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], device=pc_dep_b.device)
          dist_thresh = get_dist_thresh(calib, ct, dim, alpha)
          dist_thresh += dist_thresh * opt.frustumExpansionRatio
          pc_dep_to_hm_torch(pc_hm[i], pc_dep_b, depth, bbox, dist_thresh, opt)
      return pc_hm
```
</details>

其中的关键步骤是： `pc_dep_to_hm_torch(pc_hm[i], pc_dep_b, depth, bbox, dist_thresh, opt)`

<details>
  <summary>view code</summary>

```python
def pc_dep_to_hm_torch(pc_hm, pc_dep, dep, bbox, dist_thresh, opt):
    if isinstance(dep, list) and len(dep) > 0:
      dep = dep[0]
    ct = torch.tensor(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=torch.float32)
    # bbox_int = torch.tensor([torch.floor(bbox[0]), 
    #                      torch.floor(bbox[1]), 
    #                      torch.ceil(bbox[2]), 
    #                      torch.ceil(bbox[3])], dtype=torch.int32)# format: xyxy
    bbox_int = torch.tensor([int(torch.floor(bbox[0])), 
                         int(torch.floor(bbox[1])), 
                         int(torch.ceil(bbox[2])), 
                         int(torch.ceil(bbox[3]))], dtype=torch.int32)# format: xyxy

    roi = pc_dep[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
    pc_dep = roi[opt.pc_feat_channels['pc_dep']]
    pc_vx = roi[opt.pc_feat_channels['pc_vx']]
    pc_vz = roi[opt.pc_feat_channels['pc_vz']]

    pc_dep.sum().data
    nonzero_inds = torch.nonzero(pc_dep, as_tuple=True)
    
    if len(nonzero_inds) and len(nonzero_inds[0]) > 0:
    #   nonzero_pc_dep = torch.exp(-pc_dep[nonzero_inds])
      nonzero_pc_dep = pc_dep[nonzero_inds]
      nonzero_pc_vx = pc_vx[nonzero_inds]
      nonzero_pc_vz = pc_vz[nonzero_inds]

      ## Get points within dist threshold
      within_thresh = (nonzero_pc_dep < dep+dist_thresh) \
              & (nonzero_pc_dep > max(0, dep-dist_thresh))
      pc_dep_match = nonzero_pc_dep[within_thresh]
      pc_vx_match = nonzero_pc_vx[within_thresh]
      pc_vz_match = nonzero_pc_vz[within_thresh]

      if len(pc_dep_match) > 0:
        arg_min = torch.argmin(pc_dep_match)
        dist = pc_dep_match[arg_min]
        vx = pc_vx_match[arg_min]
        vz = pc_vz_match[arg_min]
        if opt.normalize_depth:
          dist /= opt.max_pc_dist

        w = bbox[2] - bbox[0]
        w_interval = opt.hm_to_box_ratio*(w)
        w_min = int(ct[0] - w_interval/2.)
        w_max = int(ct[0] + w_interval/2.)
        
        h = bbox[3] - bbox[1]
        h_interval = opt.hm_to_box_ratio*(h)
        h_min = int(ct[1] - h_interval/2.)
        h_max = int(ct[1] + h_interval/2.)

        pc_hm[opt.pc_feat_channels['pc_dep'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = dist
        pc_hm[opt.pc_feat_channels['pc_vx'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vx
        pc_hm[opt.pc_feat_channels['pc_vz'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vz
```
</details>

该流程就和推理阶段一样，流程如下：
1. 根据box生成像素范围
2. 根据像素范围筛选pc_feat
3. 找到范围内最近pc_dep最近的索引
4. 获取对应的'pc_dep'、'pc_vx'、'pc_vz'生成pc_hm特征


## 参考链接
- [CenterFusion 项目网络架构详细论述](https://blog.csdn.net/ssj925319/article/details/124669234)
  