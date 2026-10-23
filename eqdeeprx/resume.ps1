[CmdletBinding()]
param(
    [string]$RunDir = 'D:\EqDeepRxRuns\strict_fig6a',
    [string]$PythonPath = 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) {
    throw "Run directory was not found: $RunDir"
}
$eqAttempt = '{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID
$eqStdout = Join-Path $RunDir "train.resume.$eqAttempt.stdout.log"
$eqStderr = Join-Path $RunDir "train.resume.$eqAttempt.stderr.log"
& (Join-Path $PSScriptRoot 'train.ps1') -Resume -RunDir $RunDir -PythonPath $PythonPath 1> $eqStdout 2> $eqStderr
$eqExitCode = $LASTEXITCODE
exit $eqExitCode
