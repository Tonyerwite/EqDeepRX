param(
    [string]$OutputDir = 'D:\EqDeepRxRuns\sensitivity\matrix_screen5_full_v2'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$summaryPath = Join-Path $OutputDir 'sensitivity_summary.json'
$logPath = Join-Path $OutputDir 'supervisor.log'
$runIndex = 0

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Write-SupervisorLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

while ($true) {
    $complete = $false
    if (Test-Path -LiteralPath $summaryPath) {
        try {
            $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $complete = ($summary.execution_complete -eq $true -and $summary.status -eq 'complete')
        }
        catch {
            Write-SupervisorLog "summary read failed: $($_.Exception.Message)"
        }
    }
    if ($complete) {
        Write-SupervisorLog 'matrix complete; supervisor exiting'
        break
    }

    $existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match 'run_unpublished_sensitivity\.py' -and
            $_.CommandLine -match [regex]::Escape($OutputDir)
        }
    if ($existing) {
        Write-SupervisorLog ("existing worker detected: " + (($existing | Select-Object -ExpandProperty ProcessId) -join ','))
        Start-Sleep -Seconds 30
        continue
    }

    $runIndex++
    $stdout = Join-Path $OutputDir ("supervisor_run_{0}.stdout.log" -f $runIndex)
    $stderr = Join-Path $OutputDir ("supervisor_run_{0}.stderr.log" -f $runIndex)
    $arguments = @(
        'scripts/run_unpublished_sensitivity.py',
        '--variable', 'all',
        '--backend', 'sionna',
        '--device', 'cuda',
        '--screening-steps', '20',
        '--confirmation-steps', '20',
        '--seeds', '2026,2027',
        '--confirmation-seed', '2028',
        '--validation-samples', '400',
        '--min-in-range-samples', '1',
        '--min-in-range-per-pilot', '1',
        '--batch-size', '8',
        '--microbatch-size', '4',
        '--generation-batch-size', '2',
        '--n-layers', '4',
        '--evaluation-batch-size', '2',
        '--output-dir', $OutputDir,
        '--resume'
    )
    Write-SupervisorLog "starting worker run $runIndex"
    $worker = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    try {
        Wait-Process -Id $worker.Id -Timeout 3600 -ErrorAction SilentlyContinue
    }
    catch {
        Write-SupervisorLog "wait failed: $($_.Exception.Message)"
    }
    $stillRunning = Get-Process -Id $worker.Id -ErrorAction SilentlyContinue
    if ($stillRunning) {
        Write-SupervisorLog "worker exceeded one-hour guard; leaving it for the next supervisor pass"
        continue
    }
    $exitCode = $worker.ExitCode
    Write-SupervisorLog "worker run $runIndex exited with code $exitCode; resuming"
    Start-Sleep -Seconds 5
}
