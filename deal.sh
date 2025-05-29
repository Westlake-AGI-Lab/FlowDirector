#!/bin/bash

# --- 配置参数 ---
FPS=16              # 输出 GIF 的帧率 (frames per second)
SCALE_WIDTH=832     # 输出 GIF 的宽度 (高度会自动按比例缩放, -1 表示自动)
# SCALE_WIDTH=-1    # 如果你想指定高度并让宽度自动缩放，可以设置 SCALE_WIDTH=-1 SCALE_HEIGHT=320
# SCALE_HEIGHT=320
# OUTPUT_SUBDIR="gifs_output" # 可选：将 GIF 输出到子目录

# --- 创建输出子目录 (如果指定了) ---
if [ ! -z "$OUTPUT_SUBDIR" ]; then
  mkdir -p "$OUTPUT_SUBDIR"
  echo "GIFs will be saved in ./${OUTPUT_SUBDIR}/"
fi

# --- 遍历所有 .mp4 文件 ---
for f in assets/demos/*.mp4; do
  if [ -f "$f" ]; then # 检查文件是否存在
    filename=$(basename -- "$f")
    extension="${filename##*.}"
    filename_no_ext="${filename%.*}"

    if [ ! -z "$OUTPUT_SUBDIR" ]; then
      output_gif="${OUTPUT_SUBDIR}/${filename_no_ext}.gif"
    else
      output_gif="${filename_no_ext}.gif"
    fi

    echo "Processing '$f' -> '$output_gif'..."

    # FFmpeg 命令分解:
    # 1. palettegen: 生成一个优化的调色板，以获得更好的 GIF 质量
    # 2. paletteuse: 使用生成的调色板来创建 GIF
    # -y: 覆盖已存在的输出文件而不询问
    # -vf "fps=${FPS},scale=${SCALE_WIDTH}:-1:flags=lanczos": 设置帧率, 缩放 (lanczos 是高质量缩放算法)
    # 如果指定了 SCALE_HEIGHT 而不是 SCALE_WIDTH: -vf "fps=${FPS},scale=-1:${SCALE_HEIGHT}:flags=lanczos"

    ffmpeg -i "$f" -vf "fps=${FPS},scale=${SCALE_WIDTH}:-1:flags=lanczos,palettegen" -y /tmp/palette.png
    ffmpeg -i "$f" -i /tmp/palette.png -lavfi "fps=${FPS},scale=${SCALE_WIDTH}:-1:flags=lanczos[x]; [x][1:v]paletteuse" -y "$output_gif"

    echo "'$output_gif' created."
  fi
done

# 清理临时调色板文件
rm -f /tmp/palette.png

echo "All .mp4 files processed."