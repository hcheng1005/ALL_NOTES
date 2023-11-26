# TransFusion

## OpenPCDet模型
```yaml
MODEL:
    NAME: TransFusion

    VFE:
        NAME: MeanVFE

    BACKBONE_3D:
        NAME: VoxelResBackBone8x
        USE_BIAS: False

    MAP_TO_BEV:
        NAME: HeightCompression
        NUM_BEV_FEATURES: 256

    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]
        USE_CONV_FOR_NO_STRIDE: True


    DENSE_HEAD:
        CLASS_AGNOSTIC: False
        NAME: TransFusionHead

        USE_BIAS_BEFORE_NORM: False

        NUM_PROPOSALS: 200
        HIDDEN_CHANNEL: 128
        NUM_CLASSES: 10
        NUM_HEADS: 8
        NMS_KERNEL_SIZE: 3
        FFN_CHANNEL: 256
        DROPOUT: 0.1
        BN_MOMENTUM: 0.1
        ACTIVATION: relu

        NUM_HM_CONV: 2
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center', 'height', 'dim', 'rot', 'vel']
            HEAD_DICT: {
                'center': {'out_channels': 2, 'num_conv': 2},
                'height': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
                'vel': {'out_channels': 2, 'num_conv': 2},
            }
    
        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 8
            DATASET: nuScenes
            GAUSSIAN_OVERLAP: 0.1
            MIN_RADIUS: 2
            HUNGARIAN_ASSIGNER:
                cls_cost: {'gamma': 2.0, 'alpha': 0.25, 'weight': 0.15}
                reg_cost: {'weight': 0.25}
                iou_cost: {'weight': 0.25}
      
        LOSS_CONFIG:
            LOSS_WEIGHTS: {
                    'cls_weight': 1.0,
                    'bbox_weight': 0.25,
                    'hm_weight': 1.0,
                    'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
                }
            LOSS_CLS:
                use_sigmoid: True
                gamma: 2.0
                alpha: 0.25
          
        POST_PROCESSING:
            SCORE_THRESH: 0.0
            POST_CENTER_RANGE: [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
```

从模型配置可以看出，**TransFusion（Lidar Only）与CenterPoint的区别就只在于head的区别**。


## TransFusionHead
```python
  (dense_head): TransFusionHead(
    (loss_cls): SigmoidFocalClassificationLoss()
    (loss_bbox): L1Loss()
    (loss_heatmap): GaussianFocalLoss()
    (shared_conv): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (heatmap_head): Sequential(
      (0): BasicBlock2D(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): Conv2d(128, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (class_encoding): Conv1d(10, 128, kernel_size=(1,), stride=(1,))
    (decoder): TransformerDecoderLayer(
      (self_attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
      )
      (multihead_attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
      )
      (linear1): Linear(in_features=128, out_features=256, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (linear2): Linear(in_features=256, out_features=128, bias=True)
      (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (norm3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0.1, inplace=False)
      (dropout2): Dropout(p=0.1, inplace=False)
      (dropout3): Dropout(p=0.1, inplace=False)
      (self_posembed): PositionEmbeddingLearned(
        (position_embedding_head): Sequential(
          (0): Conv1d(2, 128, kernel_size=(1,), stride=(1,))
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (cross_posembed): PositionEmbeddingLearned(
        (position_embedding_head): Sequential(
          (0): Conv1d(2, 128, kernel_size=(1,), stride=(1,))
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
      )
    )
    (prediction_head): SeparateHead_Transfusion(
      (center): Sequential(
        (0): Sequential(
          (0): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Conv1d(64, 2, kernel_size=(1,), stride=(1,))
      )
      (height): Sequential(
        (0): Sequential(
          (0): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
      )
      (dim): Sequential(
        (0): Sequential(
          (0): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Conv1d(64, 3, kernel_size=(1,), stride=(1,))
      )
      (rot): Sequential(
        (0): Sequential(
          (0): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Conv1d(64, 2, kernel_size=(1,), stride=(1,))
      )
      (vel): Sequential(
        (0): Sequential(
          (0): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Conv1d(64, 2, kernel_size=(1,), stride=(1,))
      )
      (heatmap): Sequential(
        (0): Sequential(
          (0): Conv1d(128, 64, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Conv1d(64, 10, kernel_size=(1,), stride=(1,))
      )
    )
  )

```