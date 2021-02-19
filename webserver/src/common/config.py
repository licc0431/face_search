import os
MILVUS_HOST = os.getenv("MILVUS_HOST", "5.5.5.26")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 128)  # 创建集合的特征维度
DATA_PATH = os.getenv("DATA_PATH", "/home/lichengchao/data/face_library")  # 库的路径
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
UPLOAD_PATH="/home/lichengchao/data/search-images" # 搜索图片路径

# 特征提取模型配置
ctx_id = 0  # gpu id
model_prefix = '/home/lichengchao/face_search/webserver/src/preprocessor/models/mobilefacenet-v1/model'
model_epoch = 0000