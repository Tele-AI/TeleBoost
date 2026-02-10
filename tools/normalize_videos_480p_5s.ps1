param(
  [Parameter(Mandatory = $false)]
  [string]$Root = "videos",

  [Parameter(Mandatory = $false)]
  [int]$Height = 480,

  [Parameter(Mandatory = $false)]
  [int]$DurationSeconds = 5,

  [Parameter(Mandatory = $false)]
  [int]$Fps = 30,

  [Parameter(Mandatory = $false)]
  [int]$Crf = 23,

  [Parameter(Mandatory = $false)]
  [string]$Preset = "veryfast",

  # If true, detect a "freeze" at the end (e.g., last-frame padding) and time-stretch
  # the pre-freeze segment to DurationSeconds. This is useful when some inputs were
  # previously padded to 5s by cloning the last frame.
  [Parameter(Mandatory = $false)]
  [bool]$DetectFreeze = $true
)

$ErrorActionPreference = "Stop"

function Resolve-Exe([string]$name) {
  $wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\$name"
  if (Test-Path $wingetLinks) { return $wingetLinks }

  $cmd = Get-Command $name -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) { return $cmd.Source }

  throw "Cannot find $name. Install FFmpeg or ensure '$name' is on PATH."
}

$ffmpeg = Resolve-Exe "ffmpeg.exe"
$ffprobe = Resolve-Exe "ffprobe.exe"

if (!(Test-Path $Root)) {
  throw "Root path not found: $Root"
}

$files = Get-ChildItem -Path $Root -Recurse -File -Filter "*.mp4" | Sort-Object FullName
if (!$files -or $files.Count -eq 0) {
  Write-Host "No .mp4 files found under '$Root'."
  exit 0
}

Write-Host ("Found {0} mp4 files under '{1}'." -f $files.Count, $Root)
Write-Host ("Target: {0}p height, {1}s duration, {2}fps (H.264 CRF {3}, preset {4})." -f $Height, $DurationSeconds, $Fps, $Crf, $Preset)
Write-Host ""

$durationTarget = [double]$DurationSeconds

function Get-VideoDurationSeconds([string]$path) {
  $out = & $ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 $path
  if (!$out) { return $null }
  $d = 0.0
  if ([double]::TryParse(($out | Select-Object -First 1), [ref]$d)) { return $d }
  return $null
}

function Get-FreezeStartSeconds([string]$path) {
  if (-not $DetectFreeze) { return $null }
  # freezedetect logs lines like: "freeze_start: 3.280000"
  # Use a low noise threshold and require a minimum freeze duration.
  # Run through cmd.exe to avoid PowerShell treating stderr as a terminating error.
  $cmdLine = "`"$ffmpeg`" -hide_banner -loglevel info -i `"$path`" -an -vf `"freezedetect=n=0.003:d=0.5`" -f null - 2>&1"
  $log = & cmd /c $cmdLine
  if (!$log) { return $null }
  foreach ($line in $log) {
    if ($line -match "freeze_start:\s*(\d+(\.\d+)?)") {
      $fs = 0.0
      if ([double]::TryParse($Matches[1], [ref]$fs)) { return $fs }
    }
  }
  return $null
}

$i = 0
foreach ($f in $files) {
  $i++
  $inPath = $f.FullName
  $tmpPath = "$inPath.__tmp.mp4"

  # Skip files that are already normalized? We keep this conservative and always rewrite
  # to guarantee consistent length and resolution across all cases.

  Write-Host ("[{0}/{1}] Normalizing: {2}" -f $i, $files.Count, $inPath)

  if (Test-Path $tmpPath) { Remove-Item -Force $tmpPath }

  # We drop audio (-an) since the web demo plays muted and this avoids A/V padding edge cases.
  # If the source clip is shorter than DurationSeconds, we time-stretch (slow motion) to match
  # instead of freezing the last frame.

  $dur = Get-VideoDurationSeconds $inPath
  if ($null -eq $dur -or $dur -le 0.01) {
    Write-Host "  Skipping: could not read duration."
    continue
  }

  # If the input already contains last-frame padding, detect where motion stops and stretch only
  # the moving portion. This fixes the "stuck last frame" effect.
  $freezeStart = Get-FreezeStartSeconds $inPath
  $effectiveDur = $dur
  if ($freezeStart -and $freezeStart -gt 0.4 -and $freezeStart -lt ($dur - 0.2)) {
    $effectiveDur = $freezeStart
    Write-Host ("  Detected freeze at {0:n2}s (duration {1:n2}s). Stretching motion segment." -f $freezeStart, $dur)
  }

  # Decide time scaling: if effective segment is shorter, slow down; if longer, keep speed and trim.
  $timeScale = 1.0
  if ($effectiveDur -lt $durationTarget) {
    $timeScale = $durationTarget / $effectiveDur
  }

  # Use ${var} to avoid PowerShell interpreting patterns like $Height:flags as a scoped variable.
  # Order: trim (if needed) -> setpts (time stretch) -> scale -> fps -> format
  $trim = "trim=duration=${effectiveDur}"
  $setpts = "setpts=PTS*${timeScale}"
  $scale = "scale=-2:${Height}:flags=lanczos"
  $fpsf = "fps=${Fps}"
  $fmt = "format=yuv420p"
  $vf = "$trim,$setpts,$scale,$fpsf,$fmt"

  & $ffmpeg `
    -hide_banner -loglevel error `
    -y -i $inPath `
    -an `
    -vf $vf `
    -t $DurationSeconds `
    -c:v libx264 -preset $Preset -crf $Crf `
    -movflags +faststart `
    $tmpPath

  if (!(Test-Path $tmpPath)) {
    throw "FFmpeg did not produce output for: $inPath"
  }

  Move-Item -Force $tmpPath $inPath
}

Write-Host ""
Write-Host "Done."
