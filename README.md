# PruneMaster-HRNet 

### COCO test2017 结果
|      模型       | 参数(剪枝后)  | GFLOPS(剪枝后)  | AP   | AP.5 | AP.75 | AP M  | AP L  | AR   |
|----------------|--------------|--------------- |------|------|-------|-------|-------|------|
| HRNet-W48      | 63.6m        | 32.9           | 75.5 | 92.5 | 83.3  | 71.9  | 81.5  | 80.5 |
| APRC-HRNet-W48 | 19.7m(69.0%) | 9.8(70.3%)     | 74.6 | 92.4 | 82.4  | 71.0  | 80.6  | 79.8 |

## 环境
代码是在 Centos7 上使用 Python 3.6 开发的。需要 NVIDIA GPUs。该代码是使用 2 张 NVIDIA 2080Ti GPU 卡开发和测试的。

## 快速开始
### 安装
1. 按照[官方指南](https://pytorch.org/)安装 PyTorch >= v1.0.0。
2. clone此仓库，我们将clone的目录命名为 `${POSE_ROOT}`。
3. 安装依赖：
   ```bash
   pip install -r requirements.txt

   ```
4. 编译 libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. 安装  [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # 安装到全局 site-packages
   make install
   # 如果你没有权限或者不想安装到全局 site-packages
   python3 setup.py install --user
   ```
6. 初始化日志目录:

   ```
   mkdir log
   ```

   你的目录结构应该如下所示:

   ```
   ${POSE_ROOT}
   ├── experiments
   ├── lib
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

7. 下载原始 HRNet 的预训练模型，可以从 ([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ)) 获取：
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- hrnet_w32-36af842e.pth
            |   |-- hrnet_w48-8ef0771d.pth
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet101-5d3b4d8f.pth
            |   `-- resnet152-b121ed2d.pth
            |-- pose_coco
            |   |-- pose_hrnet_w32_256x192.pth
            |   |-- pose_hrnet_w32_384x288.pth
            |   |-- pose_hrnet_w48_256x192.pth
            |   |-- pose_hrnet_w48_384x288.pth
            |   |-- pose_resnet_101_256x192.pth
            |   |-- pose_resnet_101_384x288.pth
            |   |-- pose_resnet_152_256x192.pth
            |   |-- pose_resnet_152_384x288.pth
            |   |-- pose_resnet_50_256x192.pth
            |   `-- pose_resnet_50_384x288.pth
            `-- pose_mpii
                |-- pose_hrnet_w32_256x256.pth
                |-- pose_hrnet_w48_256x256.pth
                |-- pose_resnet_101_256x256.pth
                |-- pose_resnet_152_256x256.pth
                `-- pose_resnet_50_256x256.pth
   ```
   
我们的示例剪枝版本存在models/key_save

对于剪枝模型，主要有两个文件:
   ```
   pruneXXX.txt // 用于构建模型
   XXXXXXXX.pth // 模型权重
   ```
我们首先使用 pruneXXX.txt 来获取模型结构，然后从 XXXXXXXX.pth 复制权重。

### 数据准备
**对于 COCO 数据**，请从 [COCO下载](http://cocodataset.org/#download) 下载，需要 2017 训练/验证集进行 COCO 关键点训练和验证。我们还提供了 COCO val2017 和 test-dev2017 的人物检测结果，以复现我们的多人姿态估计结果。请从 [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) 或 [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing) 下载。
下载并解压到 {POSE_data} 下，目录结构应如下所示：
```
${POSE_data}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
### 剪枝选择和重新训练
#### 在 COCO train2017 数据集上选择剪枝
1. 编辑配置文件。例如 w48_384x288_adam_lr1e-3.yaml，
```
GPUS: (0,1)
OUTPUT_DIR: 'output'
LOG_DIR: 'log'
DATASET:
  COLOR_RGB: true
  DATASET: 'coco'
  DATA_FORMAT: jpg
  FLIP: true
  NUM_JOINTS_HALF_BODY: 8
  PROB_HALF_BODY: 0.3
  ROOT: '/root/work/datasets/coco'
  ROT_FACTOR: 45
  SCALE_FACTOR: 0.35
  TEST_SET: 'val2017'
  TRAIN_SET: 'train2017'

  PRETRAINED: 'models/pose_coco/pose_hrnet_w48_384x288.pth'

TEST:
  BATCH_SIZE_PER_GPU: 24
  COCO_BBOX_FILE: 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
  BBOX_THRE: 1.0
  IMAGE_THRE: 0.0
  IN_VIS_THRE: 0.2
  MODEL_FILE: 'models/pose_coco/pose_hrnet_w48_384x288.pth'
  NMS_THRE: 1.0
  OKS_THRE: 0.9
  USE_GT_BBOX: true
  FLIP_TEST: true
  POST_PROCESS: true
  SHIFT_HEATMAP: true

```
2. 选择通道剪枝比例
```
python3 tools/normal_regular_select \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output\
```

#### 在 COCO train2017 数据集上重新训练
```
python3 tools/prune.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --save output --percent [you get in purning select or another float in range(0,1)] \
```
#### 在 COCO 数据集上测试
修改 experiments\coco\hrnet\w48_384x288_adam_lr1e-3_pt36.yaml 中的 "MODEL_FILE"

```
python3 test.py --ncfg [{scale or shift}{$r$}.txt]
```
([{scale or shift}{$r$}.txt] Corresponding to "MODEL_FILE" in  experiments\coco\hrnet\w48_384x288_adam_lr1e-3.yaml)

