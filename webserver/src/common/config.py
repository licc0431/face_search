import os
# milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "5.5.5.26")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
VECTOR_DIMENSION = os.getenv("VECTOR_DIMENSION", 512)  # 创建集合的特征维度
DATA_PATH = os.getenv("DATA_PATH", "/home/lichengchao/data/face_library")  # 库的路径
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
UPLOAD_PATH = "/home/lichengchao/data/search-images"  # 搜索图片路径

# 人脸配置

# 人脸识别
ctx_id = 0  # gpu id
model_prefix = '/home/lichengchao/face_search/webserver/data/models/model-r34-amf/model'
model_epoch = 0000
# 人脸检测
retinaface_root = '/home/lichengchao/face_search/webserver/data/models'  # 人脸检测模型查找路径,如无会在该路径下自动下载
