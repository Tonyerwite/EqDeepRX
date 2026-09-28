param(
    [string]$OutputDir = 'D:\EqDeepRxRuns\sensitivity\joint_v1',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$output = [System.IO.Path]::GetFullPath($OutputDir)
$logDir = [System.IO.Path]::GetDirectoryName($output)
$waitLog = Join-Path $output 'waiter.log'
$stdout = Join-Path $output 'joint.stdout.log'
$stderr = Join-Path $output 'joint.stderr.log'
New-Item -ItemType Directory -Force -Path $output | Out-Null

function Write-WaitLog([string]$Message) {
    Add-Content -LiteralPath $waitLog -Value "$(Get-Date -Format o) $Message" -Encoding UTF8
}

function Test-ExternalGpuWork {
    $evaluation = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'evaluate_uncoded_ber\.py' }
    if ($evaluation) {
        return $true
    }
    try {
        $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim())
        return $used -gt 2000
    }
    catch {
        return $true
    }
}

while (Test-ExternalGpuWork) {
    Write-WaitLog 'GPU is occupied by another evaluation or has less than 2 GiB free; waiting'
    Start-Sleep -Seconds ([Math]::Max(5, $PollSeconds))
}

Write-WaitLog 'GPU is available; starting resumable joint sensitivity matrix'
$arguments = @(
    'scripts/run_joint_sensitivity.py',
    '--profiles', 'formal_baseline,fixed100,fixed300,bias_no_correction,fixed100_bias_no_correction,fixed300_bias_no_correction',
    '--steps', '2000',
    '--seeds', '2026,2027,2028',
    '--backend', 'sionna',
    '--device', 'cuda',
    '--batch-size', '8',
    '--microbatch-size', '4',
    '--generation-batch-size', '2',
    '--validation-samples', '400',
    '--n-layers', '4',
    '--evaluation-batch-size', '2',
    '--min-in-range-samples', '100',
    '--min-in-range-per-pilot', '20',
    '--output-dir', $output,
    '--resume'
)
$worker = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Wait-Process -Id $worker.Id
Write-WaitLog "joint matrix process exited with code $($worker.ExitCode)"
exit $worker.ExitCode
