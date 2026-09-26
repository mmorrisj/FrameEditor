<#
.SYNOPSIS
  Stop FrameKit starting at sign-in, and stop the copy running now.
#>
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$lnk = Join-Path ([Environment]::GetFolderPath("Startup")) "FrameKit.lnk"
if (Test-Path $lnk) { Remove-Item $lnk; Write-Host "Removed $lnk" } else { Write-Host "No startup shortcut found." }

# stop background FrameKit servers started from this folder
$procs = Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -like "*-m framekit serve*" }
foreach ($p in $procs) {
    Stop-Process -Id $p.ProcessId -Force
    Write-Host "Stopped FrameKit (process $($p.ProcessId))"
}
