# config/config.py

train_args = {
    "model_name": "bert-base-uncased",  # 使用的基础模型
    "max_length": 128,                  # 最大句子长度
    "batch_size": 16,                   # 批大小
    "lr": 2e-5,                         # 学习率
    "num_epochs": 30,                   # 最大训练轮次
    "patience": 5,                      # EarlyStopping容忍步数
    "save_path": "./checkpoints/model_best.pth",    # 保存模型的路径
    "log_path": "./checkpoints/train_log.csv",      # 保存训练日志的路径
    "device": "cuda"  # 使用cuda还是cpu
}
