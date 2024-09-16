# Counter Strike 2 players detector

#### Supported Labels
['CT', 'CT_head', 'T', 'T_head']

#### models
YOLOv10s

#### How to use
```
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO(r'weights\yolov10s_cs2.pt')

# Run inference on 'image.png' with arguments
model.predict(
    'image.png',
    save=True,
    device=0
    )
```

#### Labels
![labels.jpg](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/KRuK-Y9uEP2Hwat5ojD1E.jpeg)
#### Results
![results.png](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/-DOb5ZmGoI_vXs7zgtFMP.png)
#### Predict
![train_batch0.jpg](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/Ie7m1EQosL87TbN_UoS-0.jpeg)
![train_batch1.jpg](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/Lr3solcPWqHrdMvQ0hBX9.jpeg)
```
YOLOv10s summary (fused): 293 layers, 8,038,056 parameters, 0 gradients, 24.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s]
                   all        160        372      0.958       0.94      0.979      0.772
               ct_body         88        110      0.964      0.964      0.988      0.861
               ct_head         82        104      0.946      0.847      0.953      0.634
                t_body         70         84      0.986      0.976       0.99      0.866
                t_head         62         74      0.938      0.973      0.984      0.728
```

#### Others models Counter Strike 2 YOLOv10m Object Detection
https://huggingface.co/ChitoParedes/cs2-yolov10m
