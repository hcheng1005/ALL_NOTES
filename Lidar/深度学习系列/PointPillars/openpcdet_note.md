# OpenpcDet--PointPillars

## 参考链接
[openpcdet之pointpillar代码阅读——第一篇：数据增强与数据处理](https://blog.csdn.net/QLeelq/article/details/118807574?csdn_share_tail=%7B%22type%22:%22blog%22,%22rType%22:%22article%22,%22rId%22:%22118807574%22,%22source%22:%22QLeelq%22%7D&ctrtid=9zMqH)

[openpcdet之pointpillar代码阅读——第二篇：网络结构](https://blog.csdn.net/QLeelq/article/details/117328660?csdn_share_tail=%7B%22type%22:%22blog%22,%22rType%22:%22article%22,%22rId%22:%22117328660%22,%22source%22:%22QLeelq%22%7D&ctrtid=tcXR3)

[openpcdet之pointpillar代码阅读——第三篇：损失函数的计算](https://blog.csdn.net/QLeelq/article/details/116640084?csdn_share_tail=%7B%22type%22:%22blog%22,%22rType%22:%22article%22,%22rId%22:%22116640084%22,%22source%22:%22QLeelq%22%7D&ctrtid=BTqLQ)

## 模型结构

```ymal
VFE:
    NAME: PillarVFE
    WITH_DISTANCE: False
    USE_ABSLOTE_XYZ: True
    USE_NORM: True
    NUM_FILTERS: [64]

MAP_TO_BEV:
    NAME: PointPillarScatter
    NUM_BEV_FEATURES: 64

BACKBONE_2D:
    NAME: BaseBEVBackbone
    LAYER_NUMS: [3, 5, 5]
    LAYER_STRIDES: [2, 2, 2]
    NUM_FILTERS: [64, 128, 256]
    UPSAMPLE_STRIDES: [1, 2, 4]
    NUM_UPSAMPLE_FILTERS: [128, 128, 128]

DENSE_HEAD:
    NAME: AnchorHeadSingle
    CLASS_AGNOSTIC: False

    USE_DIRECTION_CLASSIFIER: True
    DIR_OFFSET: 0.78539
    DIR_LIMIT_OFFSET: 0.0
    NUM_DIR_BINS: 2
```


### pillarVFE
<font color="red">**功能：这部分是简化版的pointnet网络，将经过数据增强和数据处理过后的pillar(N,4)数据，经过BN层、Relu激活层和max pool层得到(C, H, W)数据。**</font>

在VFE之前的data_dict的数据如下所示：
```python
'''
batch_dict:
        points:(N,5) --> (batch_index,x,y,z,r) batch_index代表了该点云数据在当前batch中的index
        frame_id:(batch_size,) -->帧ID-->我们存放的是npy的绝对地址，batch_size个地址
        gt_boxes:(batch_size,N,8)--> (x,y,z,dx,dy,dz,ry,class)，
        use_lead_xyz:(batch_size,) --> (1,1,1,1)，batch_size个1
        voxels:(M,32,4) --> (x,y,z,r)
        voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
        voxel_num_points:(M,):每个voxel内的点云
        batch_size：batch_size大小
'''
```

随后经过VFE之后，就可以把原始的点云结构（N ∗ 4）变换成了(D， P，N)。**其中D代表了每个点云的特征维度，也就是每个点云10个特征(论文中只有9维)，P代表了所有非空的立方柱体，N代表了每个pillar中最多会有多少个点**。具体操作以及说明如下：

D ( x, y, z, xc, r, yc, zc, xp, yp, zp) ：xyz表示点云的真实坐标，下标c代表了每个点云到该点所对应pillar中所有点平均值的偏移量，下标p表示该点距离所在pillar中心点的偏移量。

P：代表了所有非空的立方柱体，yaml配置中有最大值MAX_NUMBER_OF_VOXELS。

N：代表了每个pillar中最多会有多少个点，实际操作取32。

得到(D，P，N)的张量后，接下来这里使用了一个简化版的pointnet网络对点云的数据进行特征提取（即将这些点通过MLP升维，然后跟着BN层和Relu激活层），得到一个(C，P，N)形状的张量，之后再使用max pooling操作提取每个pillar中最能代表该pillar的点。那么输出会变成(C，P，N)−>(C，P)−>(C,H,W)。

<details>
  <summary>view code</summary>
  
代码如下：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        
        # x的维度由（M, 32, 10）升维成了（M, 32, 64）,max pool之后32才去掉
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        #permute变换维度，(M, 64, 32) --> (M, 32, 64)
          # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # 完成pointnet的最大池化操作，找出每个pillar中最能代表该pillar的点
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # num_point_features:10
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        #[64]
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        # num_filters:  [10, 64]
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        #len(num_filters) - 1 == 1
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i] # 10
            out_filters = num_filters[i + 1] # 64
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        #收集PFN层，在forward中执行
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        '''
        指出一个pillar中哪些是真实数据,哪些是填充的0数据
        '''
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        '''
	batch_dict:
            points:(N,5) --> (batch_index,x,y,z,r) batch_index代表了该点云数据在当前batch中的index
            frame_id:(batch_size,) -->帧ID-->我们存放的是npy的绝对地址，batch_size个地址
            gt_boxes:(batch_size,N,8)--> (x,y,z,dx,dy,dz,ry,class)，
            use_lead_xyz:(batch_size,) --> (1,1,1,1)，batch_size个1
            voxels:(M,32,4) --> (x,y,z,r)
            voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
            voxel_num_points:(M,):每个voxel内的点云
            batch_size:4：batch_size大小
        '''
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        #求每个pillar中所有点云的平均值,设置keepdim=True的，则保留原来的维度信息
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        #每个点云数据减去该点对应pillar的平均值，得到差值 xc,yc,zc
        f_cluster = voxel_features[:, :, :3] - points_mean

        # 创建每个点云到该pillar的坐标中心点偏移量空数据 xp,yp,zp
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        '''
          coords是每个网格点的坐标，即[432, 496, 1]，需要乘以每个pillar的长宽得到点云数据中实际的长宽（单位米）
          同时为了获得每个pillar的中心点坐标，还需要加上每个pillar长宽的一半得到中心点坐标
          每个点的x、y、z减去对应pillar的坐标中心点，得到每个点到该点中心点的偏移量
        '''
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        #配置中使用了绝对坐标，直接组合即可。
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center] #10个特征，直接组合
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        #距离信息，False
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        #mask中指明了每个pillar中哪些是需要被保留的数据
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        #由0填充的数据，在计算出现xc,yc,zc和xp,yp,zp时会有值
        #features中去掉0值信息。
        features *= mask
        #执行上面收集的PFN层，每个pillar抽象出64维特征
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
```
</details>

### PointPillarScatter
<font color="red">**功能：将得到的pillar数据，投影至二维坐标中。**</font>

在经过简化版的pointnet网络提取出每个pillar的特征信息后，就需要将每个的pillar数据重新放回原来的坐标中，也就是二维坐标，组成 伪图像 数据。

对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中。

这部分代码在：`pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py`，具体的注释代码如下：


<details>
  <summary>view code</summary>

```python
import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES #64
        self.nx, self.ny, self.nz = grid_size # [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        '''
        batch_dict['pillar_features']-->为VFE得到的数据(M, 64)
        voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
        '''
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        # 根据batch_index，获取batch_size大小
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            # 创建一个空间坐标所有用来接受pillar中的数据
            # spatial_feature 维度 (64,214272)
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx #返回mask，[True, False...]
            this_coords = coords[batch_mask, :] #获取当前的batch_idx的数
            #计算pillar的索引，该点之前所有行的点总和加上该点所在的列即可
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)  # 转换数据类型
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, 214272)
            batch_spatial_features.append(spatial_feature)

        # 在第0个维度将所有的数据堆叠在一起
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
         # reshape回原空间(伪图像)    （4, 64, 214272）--> (4, 64, 496, 432)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        #返回数据
        return batch_dict

```

</details>

### BaseBEVBackbon
<font color="red">**功能：骨干网络，提取特征**</font>

经过上面的映射操作，将原来的pillar提取最大的数值后放回到相应的坐标后，就可以得到类似于图像的数据了；只有在有pillar非空的坐标处有提取的点云数据，其余地方都是0数据，所以得到的一个（batch_size，64, 432, 496）的张量还是很稀疏的。

BACKBONE_2D的输入特征维度（batch_size，64, 432, 496），输出的特征维度为[batch_size, 384, 248, 216]。

需要说明的是，主干网络构建了下采样和上采样网络，分别为加入到了blocks和deblocks中，上采样和下采样的具体操作可查看下列代码和注释。

这部分代码在：`pcdet/models/backbones_2d/base_bev_backbone.py`，具体的注释代码如下：

<details>
  <summary>view code</summary>

```python
import numpy as np
import torch
import torch.nn as nn

class BaseBEVBackbone(nn.Module):
    # input_channels = 64
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        # 层参数
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS #[3, 5, 5]
            layer_strides = self.model_cfg.LAYER_STRIDES #[2, 2, 2]
            num_filters = self.model_cfg.NUM_FILTERS # [64, 128, 256]
        else:
            layer_nums = layer_strides = num_filters = []

        # 上采样参数
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS #[128, 128, 128]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES #  [1, 2, 4]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums) #3层
        c_in_list = [input_channels, *num_filters[:-1]] # [64, 64, 128]，(*表示取值，在列表中使用，num_filters取除了最后一个元素)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        # 开始处理3层网络
        # 通道数分别为：# (64,64)-->(64,128)-->(128,256) 
        for idx in range(num_levels):
            #需要说明的是，经过这里的层，feature map变小, (h - kernel + 2p )/s + 1
            cur_layers = [
                nn.ZeroPad2d(1),
                #layer_strides：[2, 2, 2]
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            # 根据layer_nums堆叠卷积层，网络层分别为[3, 5, 5]，输入输出的通道数不变
            #需要说明的是，经过这里的层，feature map大小不变
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            #添加下采样层
            self.blocks.append(nn.Sequential(*cur_layers))
            #上采样，upsample_strides：[1, 2, 4]
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    # ConvTranspose2d：逆卷积/转置卷积，包含两步先对原 tensor 进行 上采样，然后再进行一次常规的卷积操作
                    #上采样，每两行(列)中插入（stride-1）个零行（列）
                    #卷积操作：以输入的kernel_size, 步长始终为1（和输入的stride的值无关）。
                    # num_upsample_filters：输出通道[128, 128, 128]
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) # 128 + 128 + 128 = 384
        #下面不执行：3 > 3 False
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        #输出特征384
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features:(batch_size, 64, 496, 432)
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features #输入维度：[batch_size, 64, 496, 432]
        for i in range(len(self.blocks)):
            #下采样
            x = self.blocks[i](x)
            #下采样之后，x的shape分别为：torch.Size([batch_size, 64, 248, 216])，torch.Size([batch_size, 128, 124, 108])，torch.Size([batch_size, 256, 62, 54])
            #spatial_features的shape一直为：torch.Size([batch_size, 64, 496, 432])
            stride = int(spatial_features.shape[2] / x.shape[2]) #三次分别为2，4，8
            ret_dict['spatial_features_%dx' % stride] = x
            #上采样
            #上采样不影响x的值，上采样后的值在ups中，ups中的元素维度都是：torch.Size([batch_size, 128, 248, 216])
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        #保存结果
        #输出维度：[batch_size, 384, 248, 216]，特征更多，尺度减小为原先的一半
        data_dict['spatial_features_2d'] = x

        return data_dict

```
</details>


## DENSE_HEAD

一共有三个类别的先验框，**每个先验框都有 两个方向，分别是BEV视角下的0度和90度，每个类别的先验框只有一种尺度信息**；分别是车 [3.9, 1.6, 1.56]、人[0.8, 0.6, 1.73]、自行车[1.76, 0.6, 1.73]（单位：米）。其中Car的先验框如下所示：

```yaml
{
    'class_name': 'Car', #类别
    'anchor_sizes': [[3.9, 1.6, 1.56]], #先验框的尺寸
    'anchor_rotations': [0, 1.57], #两种角度
    'anchor_bottom_heights': [-1.78], #先验框最低点高度
    'align_center': False,
    'feature_map_stride': 2,
    'matched_threshold': 0.6,#iou，正样本阈值
    'unmatched_threshold': 0.45#iou，负样本阈值，两个阈值中间的不计算损失
},

```
在anchor匹配GT的过程中，使用的是 2D IOU 匹配方式，直接从生成的特征图也就是BEV视角进行匹配，没有考虑高度的信息。

每个anchor都需要预测7个参数，分别是(x,y,z,w,l,h,θ,cls)。
> x, y, z预测一个anchor的中心坐标在点云中的位置；
> 
> w，l，h分别预测了一个anchor的长宽高数据;
> 
> θ预测了box的旋转角度。同时，因为在角度预测时候很难区分两个完全相反的box，所以PiontPillars的检测头中还添加了对一个anchor的方向预测，这里使用了一个**基于softmax的方向分类box**的两个朝向信息。
> 
> cls为预测的类别。

这部分代码在：`pcdet/models/dense_heads/anchor_head_single.py`，具体的注释代码如下：

```python
import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    '''
    Args:
        model_cfg: AnchorHeadSingle的配置
        input_channels: 384 输入通道数
        num_class: 3
        class_names: ['Car','Pedestrian','Cyclist']
        grid_size: (432, 496, 1)
        point_cloud_range: (0, -39.68, -3, 69.12, 39.68, 1)
        predict_boxes_when_training: False
    '''
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        #在父类中调用generate_anchors中生成anchors和num_anchors_per_location
        # 每个点会生成不同类别的2个先验框(anchor)，也就是说num_anchors_per_location：[2, 2, 2,]-》3类，每类2个anchor
        #所以每个点生成6个先验框(anchor)
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        #类别， 1x1 卷积：conv_cls:  Conv2d(384, 18, kernel_size=(1, 1), stride=(1, 1))
        #每个点6个anchor，每个anchor预测3个类别，所以输出的类别为6*3
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        #box，1x1 卷积：conv_box:  Conv2d(384, 42, kernel_size=(1, 1), stride=(1, 1))
        #每个点6个anchor，每个anchor预测7个值（x, y, z, w, l, h, θ），所以输出的值为6*7
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, #self.box_coder.code_size默认为7
            kernel_size=1
        )
        # 是否使用方向分类，1x1 卷积：conv_dir:  Conv2d(384, 12, kernel_size=(1, 1), stride=(1, 1))
        #每个点6个anchor，每个anchor预测2个方向(正负)，所以输出的值为6*2
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()
    
    #初始化参数
    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # spatial_features_2d 维度 ：（batch_size, 384, 248, 216）
        spatial_features_2d = data_dict['spatial_features_2d']

        #cls_preds的维度为：torch.Size([batch_size, 18, 248, 216])
        #每个点6个anchor，每个anchor预测3个类别，所以输出的类别为6*3
        cls_preds = self.conv_cls(spatial_features_2d)
        #box_preds的维度为：torch.Size([batch_size, 42, 248, 216])
        #每个点6个anchor，每个anchor预测7个值（x, y, z, w, l, h, θ），所以输出的值为6*7
        box_preds = self.conv_box(spatial_features_2d)

        #调整顺序
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        #方向预测，也就是正负预测
        if self.conv_dir_cls is not None:
            #dir_cls_preds的维度为：torch.Size([batch_size, 12, 248, 216])
            #每个点6个anchor，每个anchor预测2个方向(正负)，所以输出的值为6*2
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            #调整顺序
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        #如果是在训练模式的时候，需要对每个先验框分配GT来计算loss
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            #分配的
            self.forward_ret_dict.update(targets_dict)
        #非训练模式，则直接生成进行box的预测
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

```