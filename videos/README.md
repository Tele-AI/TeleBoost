# Video Directory Guide

This folder stores all showcased videos.

## Structure
```
videos/
├── t2v/              # Text-to-Video
├── i2v/              # Image-to-Video
└── comparisons/      # Comparison cases
    ├── case1/
    │   ├── baseline.mp4
    │   ├── dancegrpo.mp4
    │   └── ours.mp4
    ├── case2/
    │   ├── baseline.mp4
    │   ├── dancegrpo.mp4
    │   └── ours.mp4
    └── ...
```

## How to add videos
1) Place files in the correct subfolder:  
   - T2V → `videos/t2v/`  
   - I2V → `videos/i2v/`  
   - Comparisons → `videos/comparisons/caseX/`

2) Update `videos-config.js` with entries for each video.

### Examples
Add a T2V video:
```javascript
t2v: [
  {
    title: "Sunset scene",
    description: "A sunset with waves hitting the shore",
    videoUrl: "videos/t2v/sunset.mp4"
  }
]
```

Add an I2V video:
```javascript
i2v: [
  {
    title: "Dynamic image",
    description: "Turning a static image into motion",
    videoUrl: "videos/i2v/dynamic_image.mp4"
  }
]
```

Add a comparison case:
```javascript
comparisons: [
  {
    title: "Case 1",
    prompt: "A beautiful sunset over the ocean with waves crashing on the shore",
    videos: {
      baseline: "videos/comparisons/case1/baseline.mp4",
      dancegrpo: "videos/comparisons/case1/dancegrpo.mp4",
      ours: "videos/comparisons/case1/ours.mp4"
    }
  }
]
```

## Recommendations
- Format: MP4 (H.264)
- Resolution: 1280x720 or higher
- Frame rate: 24fps or 30fps
- Keep file names simple (letters/numbers) and avoid special characters
- Large files: compress when possible
- All videos auto-loop and start muted for smoother preview
