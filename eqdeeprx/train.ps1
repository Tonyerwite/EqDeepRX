[CmdletBinding()]
param(
    [switch]$Resume,
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
    if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) {
        New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    }
    $eqReportPath = Join-Path $RunDir 'preflight.json'
    if (-not (Test-Path -LiteralPath $eqReportPath -PathType Leaf)) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'verified_run.json') -Destination $eqReportPath
    }
    $eqRuntime = Join-Path $RunDir 'runtime'
    $eqTemp = Join-Path $eqRuntime 'tmp'
    $eqMpl = Join-Path $eqRuntime 'mpl'
    New-Item -ItemType Directory -Path $eqTemp, $eqMpl -Force | Out-Null
    $env:PYTORCH_ALLOC_CONF = 'max_split_size_mb:256'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:TMP = $eqTemp
    $env:TEMP = $eqTemp
    $env:TMPDIR = $eqTemp
    $env:MPLCONFIGDIR = $eqMpl
    $eqArguments = @('-m', 'eqdeeprx', 'train', '--run-dir', $RunDir,
        '--steps', '70000', '--save-every', '50', '--confirm')
    if ($Resume) { $eqArguments += '--resume' }
    $eqPreviousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 promotes redirected native stderr to an
        # error record. Python warnings must not terminate a healthy run.
        $ErrorActionPreference = 'Continue'
        & $PythonPath @eqArguments
        $eqExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $eqPreviousErrorActionPreference
    }
}
finally {
    Pop-Location
}
exit $eqExitCode
