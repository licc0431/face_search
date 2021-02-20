# From tensorflow/tensorflow
FROM ubuntu
COPY . /app    # 复制文件到容器里指定路径
# ENV TF_XLA_FLAGS --tf_xla_cpu_global_jit     # 设置环境变量
RUN mkdir -p /root/.mxnet/models && mv /app/data/models/* /root/.mxnet/models/

RUN apt-get update && apt-get install python3-pip python3 -y
RUN pip3 install -r /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

RUN mkdir -p /tmp/search-images


#CMD gunicorn --bind 0.0.0.0:5000 -w 2 app:app --preload
WORKDIR /app/src   # WORKDIR：在容器内设置工作目录
CMD python3 app.py
