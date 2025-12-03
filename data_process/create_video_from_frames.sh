#!/bin/bash

# 从帧序列创建视频的脚本
cd /home/yinzhenhan/Human_Pre/dataset/visual_test/urdf_frames_simple/

echo "正在从帧序列创建视频..."
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p robot_urdf_fixed_pose_from_frames.mp4

if [ $? -eq 0 ]; then
    echo "✅ 视频创建成功！"
    echo "视频文件: robot_urdf_fixed_pose_from_frames.mp4"
    ls -la robot_urdf_fixed_pose_from_frames.mp4
else
    echo "❌ 视频创建失败"
fi