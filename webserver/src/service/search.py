import logging
from common.const import default_cache_dir
from common.config import ctx_id, model_prefix, model_epoch
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from diskcache import Cache
from preprocessor.face_model import FaceModel
import numpy as np

def query_name_from_ids(vids):     # 根据ID查询name
    res = []
    cache = Cache(default_cache_dir)
    for i in vids:
        if i in cache:
            res.append(cache[i])
    return res


def do_search(table_name, img_path, top_k):
    try:
        feats = []
        index_client = milvus_client()
        model = FaceModel(ctx_id, model_prefix, model_epoch)
        detection = model.get_input(img_path)       # 人脸检测，对齐，裁剪
        if detection is None:
            print("No face found in the search image, please change the search image")
        else:
            feat = model.get_feature(detection)   # 人脸特征提取
            feats.append(feat)
            feats = np.array(feats)   # feats的格式需numpy.ndarray，注意特征维度匹配
            _, vectors = search_vectors(index_client, table_name, feats, top_k)
            vids = [x.id for x in vectors[0]]
            # res = [x.decode('utf-8') for x in query_name_from_ids(vids)]
            res_id = [x.decode('utf-8') for x in query_name_from_ids(vids)]
            # print(res_id)
            res_distance = [x.distance for x in vectors[0]]
            # print(res_distance)
            # res = dict(zip(res_id,distance))
            res_similarity = [((i*0.5+0.5)*100) for i in res_distance]   # 余弦相似度归一化到[0, 1]
            return res_id, res_similarity
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)
