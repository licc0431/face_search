import logging
import time
from common.config import DEFAULT_TABLE
from common.config import ctx_id, model_prefix, model_epoch
from common.const import default_cache_dir
# from common.config import DATA_PATH as database_path
from encoder.encode import feature_extract
from diskcache import Cache
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index,has_table
from preprocessor.face_model import FaceModel


def do_train(table_name, database_path):
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(default_cache_dir)
    try:
        start_time = time.time()
        vectors, names = feature_extract(database_path, FaceModel(ctx_id, model_prefix, model_epoch))   # feature_extract-encode
        index_client = milvus_client()  # 连接服务器
        #delete_table(index_client, table_name=table_name) # 如果集合配置有变，需删除原集合或者新建
        # time.sleep(1)
        status, ok = has_table(index_client, table_name)  # 创建集合
        if not ok:
            print("create table.")
            create_table(index_client, table_name=table_name)
        print("insert into:", table_name)
        status, ids = insert_vectors(index_client, table_name, vectors)  # 插入向量
        create_index(index_client, table_name)  # 创建索引
        for i in range(len(names)):
            # cache[names[i]] = ids[i]
            cache[ids[i]] = names[i]
        total = vectors.shape[0]

        print(total)
        end_time = time.time()
        average_time = (end_time - start_time) / total
        print(average_time)
        print("Train finished")
        return "Train finished"

    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e)

