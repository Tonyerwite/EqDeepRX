[CmdletBinding()]
param(
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
trap {
    Add-Content -LiteralPath 'D:\EqDeepRxRuns\strict_fig6a\evaluate.launcher.stderr.log' -Value ($_ | Out-String)
    exit 1
}

$eqRunDir = 'D:\EqDeepRxRuns\strict_fig6a'
$eqWorkspace = Split-Path -Parent $PSScriptRoot
$eqPython = 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe'
$eqLog = Join-Path $eqRunDir 'training.jsonl'
$eqCheckpoint = Join-Path $eqRunDir 'eqdeeprx.pt'

if (-not (Test-Path -LiteralPath $eqPython -PathType Leaf)) {
    throw "CUDA Python was not found: $eqPython"
}
if (-not (Test-Path -LiteralPath $eqCheckpoint -PathType Leaf)) {
    throw "Training checkpoint was not found: $eqCheckpoint"
}
$eqLast = Get-Content -LiteralPath $eqLog -Tail 1 | ConvertFrom-Json
if ([int]$eqLast.step -ne 70000) {
    throw "Refusing Fig. 6(a) evaluation before step 70000; latest log step is $($eqLast.step)"
}

$eqStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$eqStdout = Join-Path $eqRunDir "evaluate.corrected.$eqStamp.stdout.log"
$eqStderr = Join-Path $eqRunDir "evaluate.corrected.$eqStamp.stderr.log"
$eqScript = Join-Path $eqWorkspace 'evaluate_fig6a_corrected.py'
$eqArgs = @("`"$eqScript`"",
    '--run-dir', $eqRunDir, '--samples', '32000', '--layers', '2')
if ($Resume) { $eqArgs += '--resume' }

$eqProcess = Start-Process -FilePath $eqPython -ArgumentList $eqArgs -WorkingDirectory $eqWorkspace -RedirectStandardOutput $eqStdout -RedirectStandardError $eqStderr -WindowStyle Hidden -Wait -PassThru
exit $eqProcess.ExitCode
