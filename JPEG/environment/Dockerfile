# hash:sha256:bfc410c079ac39b936c2286411af68ae4aa4bd3a461668e422038f5456cf1c1d
FROM registry.codeocean.com/codeocean/matlab:2020b-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MLM_LICENSE_FILE

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        libjpeg62-dev=1:6b2-3 \
    && rm -rf /var/lib/apt/lists/*
