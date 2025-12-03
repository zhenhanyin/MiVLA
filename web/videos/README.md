# MiVLA Project Page Videos

This directory contains video files for the MiVLA project demonstration.

## Supported Video Formats

The website supports the following video formats:
- **MP4** (recommended): `.mp4` files with H.264 codec
- **WebM**: `.webm` files with VP8/VP9 codec

## Adding Your Demo Video

### Step 1: Prepare Your Video File

1. **Format**: Convert your video to MP4 format (H.264 + AAC)
2. **Resolution**: Recommended 1920x1080 (1080p) or 1280x720 (720p)
3. **Length**: Keep it concise (1-3 minutes recommended)
4. **File Size**: Try to keep under 100MB for web optimization

### Step 2: Name Your Video

Rename your video file to:
```
mivla_demo.mp4
```

### Step 3: Add Video to Website

Place your video file in this directory:

```bash
# Example
cp your_demo_video.mp4 videos/mivla_demo.mp4
```

### Optional: Add WebM Format

For better browser compatibility, you can also add a WebM version:

```bash
# Convert to WebM (optional)
ffmpeg -i mivla_demo.mp4 -c:v libvpx-vp9 -c:a libvorbis mivla_demo.webm
```

### Optional: Add Custom Thumbnail

To add a custom thumbnail for your video:

1. Create a thumbnail image (1920x1080 recommended)
2. Name it: `mivla_demo_poster.jpg`
3. Place it in the `images/` directory:

```bash
cp your_thumbnail.jpg images/mivla_demo_poster.jpg
```

The thumbnail will be displayed before the video starts playing.

## Video Content Recommendations

Your demo video should showcase:

1. **Model Capabilities**: Various robot manipulation tasks
2. **Multi-Platform Support**: Different robots (ARX, PiPer, LocoMan)
3. **Environments**: Both simulation and real-world scenarios
4. **Task Diversity**: Different types of manipulation challenges
5. **Performance Highlights**: Successful task completions

## Technical Tips

### Video Compression

Use these FFmpeg commands for optimal web video:

```bash
# High quality MP4 (recommended)
ffmpeg -i input_video.mp4 -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k mivla_demo.mp4

# Smaller file size
ffmpeg -i input_video.mp4 -c:v libx264 -preset slow -crf 28 -c:a aac -b:a 96k mivla_demo.mp4
```

### Thumbnail

The video will use the first frame as thumbnail. To add a custom thumbnail:

1. Add a poster image: `videos/mivla_demo_poster.jpg`
2. Update the HTML video tag: `<video poster="videos/mivla_demo_poster.jpg" ...>`

## Browser Support

- **MP4**: Supported by all modern browsers
- **WebM**: Fallback format for better compression
- **Autoplay**: Disabled by default in most browsers (user must click to play)
- **Controls**: Play/pause, volume, fullscreen, and progress bar are included

## Troubleshooting

### Video Not Playing
1. Check file format (MP4 with H.264 codec recommended)
2. Verify file path and name
3. Check browser console for error messages
4. Ensure file permissions are correct

### Large File Size
1. Compress the video using lower CRF values
2. Reduce resolution if necessary
3. Consider using shorter video segments

### Mobile Compatibility
1. Test on mobile devices
2. Ensure responsive sizing works correctly
3. Consider adding mobile-specific video formats if needed