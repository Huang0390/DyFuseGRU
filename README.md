# DyFuseGRU

结构响应预测模型及训练代码。安装依赖：

```bash
pip install -r requirements.txt
```

参考 `manifest.example.json` 编写 `manifest.json`。三个 `.npy` 数组分别为地震输入、目标响应和辅助响应，形状均为 `[楼层数, 时间点数, 1]`。同一地震动的不同强度记录应使用相同的 `group_id`。辅助响应须在预测时可获得，不能由目标响应生成。

训练示例（`--floors` 填数组第一维）：

```bash
python train.py train --manifest manifest.json --floors 21 \
  --history 500 --horizon 20 --dim 512 --layers 2 \
  --epochs 100 --output runs/experiment
```

测试集评估：

```bash
python train.py evaluate --manifest manifest.json \
  --checkpoint runs/experiment/best.pt --split test
```
