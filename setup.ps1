#Requires -Version 5.1
<#
.SYNOPSIS
  Bootstraps llm-connector inside WSL2 (Windows requires WSL2 for GPU support).
.DESCRIPTION
  Converts the current Windows path to a WSL path and runs setup.sh inside WSL.
  Requires WSL2 with a Linux distribution already installed.
#>

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Error "WSL is not available."
    Write-Host ""
    Write-Host "Install WSL2 first, then re-run this script:"
    Write-Host "  wsl --install"
    Write-Host ""
    Write-Host "After WSL2 is ready, also ensure you have:"
    Write-Host "  - NVIDIA: CUDA-capable drivers for WSL2 (https://developer.nvidia.com/cuda/wsl)"
    Write-Host "  - AMD:    ROCm for WSL2 (https://rocm.docs.amd.com/en/latest/deploy/windows)"
    exit 1
}

$WslScriptPath = wsl wslpath -a ($PSScriptRoot -replace '\\', '/')
wsl -- bash -c "cd '$WslScriptPath' && bash setup.sh"
