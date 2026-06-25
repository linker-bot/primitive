#!/bin/bash

# 目标文件
TARGET_FILE="indoor_ds_new.ckpt"
# 预期文件大小（约 46.4MB，单位：字节）
EXPECTED_SIZE=48653926
# Google Drive 文件 ID
FILE_ID="19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS"
# 重试间隔（秒）
RETRY_WAIT=2

echo "开始下载模型文件..."
echo "目标文件: $TARGET_FILE"
echo "预期大小: $EXPECTED_SIZE 字节"
echo "=============================="

while true; do
    # 尝试下载（支持断点续传）
    gdown --continue "https://drive.google.com/uc?id=$FILE_ID" -O "$TARGET_FILE"
    
    # 检查文件是否存在
    if [ ! -f "$TARGET_FILE" ]; then
        echo "下载失败：文件不存在，$RETRY_WAIT 秒后重试..."
        sleep $RETRY_WAIT
        continue
    fi
    
    # 获取实际文件大小
    ACTUAL_SIZE=$(stat -c%s "$TARGET_FILE" 2>/dev/null || echo "0")
    
    # 检查文件是否完整
    if [ "$ACTUAL_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo "下载完成！文件大小: $ACTUAL_SIZE 字节"
        echo "=============================="
        exit 0
    else
        echo "下载中断：当前进度 $ACTUAL_SIZE/$EXPECTED_SIZE 字节"
        echo "$RETRY_WAIT 秒后继续下载..."
        sleep $RETRY_WAIT
    fi
done
