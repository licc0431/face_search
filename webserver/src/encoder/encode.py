import os
import numpy as np
from common.config import DATA_PATH as database_path
from encoder.utils import get_imlist
from diskcache import Cache
from common.const import default_cache_dir


def feature_extract(database_path, model):
    cache = Cache(default_cache_dir)
    feats = []
    names = []
    img_list = get_imlist(database_path)
    model = model
    for i, img_path in enumerate(img_list):
        detection = model.get_input(img_path)  # 人脸检测，对齐，裁剪
        if detection is not None:
            norm_feat = model.get_feature(detection)  # 人脸特征提取
            img_name = os.path.split(img_path)[1]
            feats.append(norm_feat)
            names.append(img_name.encode())
            current = i + 1
            total = len(img_list)
            cache['current'] = current
            cache['total'] = total
            print("extracting feature from image No. %d , %d images in total" % (current, total))
    feats = np.array(feats)  # 特征格式需numpy.ndarray，注意特征维度匹配
    return feats, names
