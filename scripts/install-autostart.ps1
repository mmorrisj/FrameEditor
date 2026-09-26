<#
.SYNOPSIS
  Start FrameKit in the background every time you sign in to Windows.

.DESCRIPTION
  Puts a shortcut in your Startup folder that runs the web app with pythonw
  (no console window). Output goes to work\server.log. No admin rights needed.
  Remove it again with scripts\uninstall-autostart.ps1.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1 -StartNow
.EXAMPLE
  # allow other machines on your network (there is no login, so only on a trusted network)
  powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1 -ListenHost 0.0.0.0
#>
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8082,
    [switch]$StartNow
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

$pythonw = Join-Path $repo "venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "pythonw.exe not found. Create the venv first: python -m venv venv; venv\Scripts\pip install -r requirements.txt" }
    $pythonw = $cmd.Source
}
if (-not (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg is not on PATH. Install it with: winget install Gyan.FFmpeg"
}

$log = Join-Path $repo "work\server.log"
$arguments = "-m framekit serve --host $ListenHost --port $Port --log `"$log`""
$lnk = Join-Path ([Environment]::GetFolderPath("Startup")) "FrameKit.lnk"

$shell = New-Object -ComObject WScript.Shell
$s = $shell.CreateShortcut($lnk)
$s.TargetPath = $pythonw
$s.Arguments = $arguments
$s.WorkingDirectory = $repo
$s.Description = "FrameKit video tools (http://localhost:$Port)"
$s.Save()
Write-Host "Installed: $lnk"
Write-Host "FrameKit will start at sign-in on http://localhost:$Port  (log: $log)"

if ($StartNow) {
    Start-Process -FilePath $pythonw -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden
    Write-Host "Started. Open http://localhost:$Port"
}
