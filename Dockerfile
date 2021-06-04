FROM python:3.8-slim
RUN mkdir /intern
WORKDIR /intern
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
VOLUME /intern/dataset
COPY . .
CMD ["bash"]

