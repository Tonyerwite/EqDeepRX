[CmdletBinding()]
param(
    [switch]$Resume,
    [ValidateSet(2, 3, 4)][int]$Layers = 4,
    [string]$RunDir = 'D:\EqDeepRxRuns\strict_fig6a',
    [string]$PythonPath = 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "CUDA Python was not found: $PythonPath"
}
$eqWorkspace = Split-Path -Parent $PSScriptRoot
$eqExitCode = 1
Push-Location -LiteralPath $eqWorkspace
try {
    $eqArguments = @('-m', 'eqdeeprx', 'evaluate', '--run-dir', $RunDir,
        '--samples', '32000', '--layers', [string]$Layers)
    if ($Resume) { $eqArguments += '--resume' }
    & $PythonPath @eqArguments
    $eqExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $eqExitCode
