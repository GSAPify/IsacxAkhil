param(
    [switch]$MonitorOnly,
    [int]$BatchSize = 32,
    [int]$Iterations = 200
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Write-Header {
    param([string]$Text)
    $line = "=" * 90
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Write-Label {
    param([string]$Label, [string]$Value)
    Write-Host "  $($Label.PadRight(18)): " -NoNewline -ForegroundColor DarkGray
    Write-Host $Value -ForegroundColor White
}

function Test-NvidiaSmi {
    try {
        $null = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Get-GpuInfo {
    $raw = & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>&1
    $parts = $raw -split ","
    if ($parts.Count -ge 8) {
        return @{
            Name        = $parts[0].Trim()
            Driver      = $parts[1].Trim()
            VRAMTotal   = $parts[2].Trim()
            VRAMFree    = $parts[3].Trim()
            Temperature = $parts[4].Trim()
            PowerLimit  = $parts[5].Trim()
            PCIeGen     = $parts[6].Trim()
            PCIeWidth   = $parts[7].Trim()
        }
    }
    return $null
}

function Show-GpuMonitor {
    param([int]$DurationSeconds = 0)
    $startTime = Get-Date
    while ($true) {
        $raw = & nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits 2>&1
        $p = ($raw -split ",") | ForEach-Object { $_.Trim() }
        if ($p.Count -ge 9) {
            $gpuUtil  = [int]$p[0]
            $memUtil  = [int]$p[1]
            $memUsed  = [int]$p[2]
            $memTotal = [int]$p[3]
            $temp     = [int]$p[4]
            $power    = $p[5]
            $powerCap = $p[6]
            $smClk    = $p[7]
            $memClk   = $p[8]
            $barLen = 25
            $gpuFill = [math]::Floor($gpuUtil / 100 * $barLen)
            $memFill = [math]::Floor($memUtil / 100 * $barLen)
            $gpuBar = ("#" * $gpuFill) + ("-" * ($barLen - $gpuFill))
            $memBar = ("#" * $memFill) + ("-" * ($barLen - $memFill))
            $tempColor = if ($temp -ge 80) { "Red" } elseif ($temp -ge 65) { "Yellow" } else { "Green" }
            Clear-Host
            Write-Host ""
            Write-Host "  GPU REAL-TIME MONITOR" -ForegroundColor Cyan
            Write-Host "  $("-" * 70)" -ForegroundColor DarkGray
            Write-Host ""
            Write-Host "  GPU Load  : [" -NoNewline
            Write-Host $gpuBar -NoNewline -ForegroundColor $(if ($gpuUtil -ge 90) { "Red" } elseif ($gpuUtil -ge 50) { "Yellow" } else { "Green" })
            Write-Host "] $($gpuUtil.ToString().PadLeft(3))%"
            Write-Host "  VRAM      : [" -NoNewline
            Write-Host $memBar -NoNewline -ForegroundColor $(if ($memUtil -ge 90) { "Red" } elseif ($memUtil -ge 50) { "Yellow" } else { "Green" })
            Write-Host "] $memUsed / $memTotal MB"
            Write-Host ""
            Write-Host "  Temp      : " -NoNewline
            Write-Host "$temp C" -ForegroundColor $tempColor
            Write-Host "  Power     : $power / $powerCap W"
            Write-Host "  SM Clock  : $smClk MHz"
            Write-Host "  Mem Clock : $memClk MHz"
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host ""
            Write-Host "  Uptime    : $([math]::Floor($elapsed))s" -ForegroundColor DarkGray
            Write-Host "  Press Ctrl+C to stop" -ForegroundColor DarkGray
        }
        if ($DurationSeconds -gt 0 -and ((Get-Date) - $startTime).TotalSeconds -ge $DurationSeconds) { break }
        Start-Sleep -Seconds 1
    }
}

function Test-PythonEnvironment {
    $venvPath = Join-Path $ProjectRoot "env_isaacsim"
    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    if (Test-Path $venvPython) {
        Write-Host "  Using venv  : env_isaacsim" -ForegroundColor Green
        return $venvPython
    }
    try {
        $pyPath = (Get-Command python -ErrorAction Stop).Source
        Write-Host "  Using python: $pyPath" -ForegroundColor Yellow
        return "python"
    } catch {
        return $null
    }
}

function Test-TorchCuda {
    param([string]$Python)
    $check = & $Python -c "import torch; print(f'{torch.cuda.is_available()}|{torch.__version__}|{torch.version.cuda if torch.cuda.is_available() else ''N/A''}')" 2>&1
    if ($LASTEXITCODE -ne 0) { return $null }
    $parts = ($check -split "\|")
    return @{
        Available = $parts[0] -eq "True"
        Version   = $parts[1]
        Cuda      = $parts[2]
    }
}

Write-Header "GPU DIAGNOSTIC & STRESS TEST"

if (-not (Test-NvidiaSmi)) {
    Write-Host "  nvidia-smi not found. Is the NVIDIA driver installed?" -ForegroundColor Red
    exit 1
}

$gpuInfo = Get-GpuInfo
if ($gpuInfo) {
    Write-Label "GPU" $gpuInfo.Name
    Write-Label "Driver" $gpuInfo.Driver
    Write-Label "VRAM" "$($gpuInfo.VRAMTotal) MB total, $($gpuInfo.VRAMFree) MB free"
    Write-Label "Temperature" "$($gpuInfo.Temperature) C"
    Write-Label "Power Limit" "$($gpuInfo.PowerLimit) W"
    Write-Label "PCIe" "Gen $($gpuInfo.PCIeGen) x$($gpuInfo.PCIeWidth)"
}

if ($MonitorOnly) {
    Write-Host ""
    Write-Host "  Monitor-only mode. No stress test will run." -ForegroundColor Yellow
    Show-GpuMonitor
    exit 0
}

Write-Header "PYTHON ENVIRONMENT CHECK"

$python = Test-PythonEnvironment
if (-not $python) {
    Write-Host "  Python not found. Install Python or create env_isaacsim venv." -ForegroundColor Red
    exit 1
}

$torchInfo = Test-TorchCuda -Python $python
if (-not $torchInfo) {
    Write-Host "  PyTorch not installed. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Red
    exit 1
}

Write-Label "PyTorch" $torchInfo.Version
Write-Label "CUDA" $torchInfo.Cuda
if (-not $torchInfo.Available) {
    Write-Host ""
    Write-Host "  CUDA not available to PyTorch!" -ForegroundColor Red
    Write-Host "  You may have CPU-only PyTorch installed." -ForegroundColor Red
    Write-Host "  Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Yellow
    exit 1
}
Write-Host "  CUDA available  : YES" -ForegroundColor Green

Write-Header "RUNNING STRESS TEST"
Write-Host "  Batch size: $BatchSize | Iterations: $Iterations" -ForegroundColor DarkGray
Write-Host ""

$stressScript = Join-Path $PSScriptRoot "gpu_stress_test.py"
if (-not (Test-Path $stressScript)) {
    Write-Host "  gpu_stress_test.py not found at $stressScript" -ForegroundColor Red
    exit 1
}

$env:GPU_TEST_BATCH_SIZE = $BatchSize
$env:GPU_TEST_ITERATIONS = $Iterations

& $python $stressScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Stress test failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "  All tests passed. GPU is ready." -ForegroundColor Green
Write-Host ""
