FROM paddlepaddle/paddle:2.6.1-gpu-cuda11.2-cudnn8.2-trt8.0

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/PaddlePaddle/Serving
RUN cd Serving
RUN pip3 install --upgrade pip
RUN pip3 install -r python/requirements.txt
RUN pip3 install paddle-serving-server-gpu==0.9.0.post1028
RUN pip3 install paddlepaddle-gpu==2.6.1

#RUN pip3 install \
#  git+https://github.com/1adrianb/face-alignment \
#  -r requirements.txt
#RUN pip3 install opencv-fixer==0.2.5
#RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"