#!/bin/bash

# 从第三视角帧序列创建视频的脚本
cd /home/yinzhenhan/Human_Pre/visual_test/high_precision_frames_basic_final-0

echo "正在从第三视角帧序列创建视频..."
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p robot_urdf_third_person_view.mp4

if [ $? -eq 0 ]; then
    echo "✅ 第三视角视频创建成功！"
    echo "视频文件: robot_urdf_third_person_view.mp4"
    ls -la robot_urdf_third_person_view.mp4
else
    echo "❌ 视频创建失败"
fi