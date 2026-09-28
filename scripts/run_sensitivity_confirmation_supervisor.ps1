param(
    [string]$ScreeningDir = 'D:\EqDeepRxRuns\sensitivity\matrix_screen5_full_v2',
    [string]$RootOutputDir = 'D:\EqDeepRxRuns\sensitivity\confirmation_v2',
    [ValidateSet('all', 'screening-winners')]
    [string]$SelectionMode = 'all',
    [int]$ConfirmationSteps = 2000,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$minConfirmationSteps = 2000
if ($ConfirmationSteps -lt $minConfirmationSteps) {
    throw "ConfirmationSteps must be at least $minConfirmationSteps"
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$logPath = Join-Path $RootOutputDir 'supervisor.log'
$manifestPath = Join-Path $RootOutputDir 'confirmation_manifest.json'

New-Item -ItemType Directory -Force -Path $RootOutputDir | Out-Null

function Write-ConfirmationLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Read-Summary([string]$Directory) {
    $path = Join-Path $Directory 'sensitivity_summary.json'
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Write-ConfirmationLog "summary read failed for $Directory`: $($_.Exception.Message)"
        return $null
    }
}

function Convert-CandidateValue($Value) {
    if ($Value -is [bool]) {
        return $Value.ToString().ToLowerInvariant()
    }
    if ($Value -is [System.IFormattable]) {
        return $Value.ToString($null, [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return $Value.ToString()
}

Write-ConfirmationLog 'waiting for the schema-5 screening matrix to become complete'
if ($DryRun) {
    $screening = Read-Summary $ScreeningDir
    if (-not ($screening -and $screening.execution_complete -eq $true -and $screening.status -eq 'complete')) {
        throw "DryRun requires a complete screening summary at $ScreeningDir"
    }
}
else {
    while ($true) {
        $screening = Read-Summary $ScreeningDir
        if ($screening -and $screening.execution_complete -eq $true -and $screening.status -eq 'complete') {
            break
        }
        Start-Sleep -Seconds 60
    }
}
Write-ConfirmationLog 'screening matrix complete; starting confirmation jobs'

# Derive the job list only after the complete matrix is available. By default,
# confirm every value in the complete candidate matrix. The former behavior of
# confirming only two-seed screening winners remains available explicitly via
# -SelectionMode screening-winners, but is not sufficient for a three-seed
# candidate audit because a 20-step screen can discard a slower-starting value.
$jobs = @()
$groups = $screening.candidate_decisions | Group-Object variable
foreach ($group in $groups) {
    $baseline = @($group.Group | Where-Object { $_.decision -eq 'baseline' })
    if ($baseline.Count -ne 1) {
        throw "expected exactly one formal baseline for $($group.Name)"
    }
    if ($SelectionMode -eq 'all') {
        $matrixProperty = @(
            $screening.candidate_matrix.PSObject.Properties |
                Where-Object { $_.Name -eq $group.Name }
        )
        if ($matrixProperty.Count -ne 1) {
            throw "candidate_matrix is missing exactly one entry for $($group.Name)"
        }
        $values = @(
            @($matrixProperty[0].Value) |
                ForEach-Object { Convert-CandidateValue $_ } |
                Select-Object -Unique
        )
    }
    else {
        $winners = @(
            $group.Group |
                Where-Object { $_.decision -eq 'screening_improvement_only' }
        )
        $values = @(
            @($baseline[0].value) +
            @($winners | ForEach-Object { $_.value }) |
                ForEach-Object { Convert-CandidateValue $_ } |
                Select-Object -Unique
        )
    }
    if ($values.Count -eq 0) {
        throw "no candidate values found for $($group.Name)"
    }
    $jobs += @{
        Name = $group.Name
        Variable = $group.Name
        Values = ($values -join ',')
    }
}
if ($jobs.Count -eq 0) {
    Write-ConfirmationLog 'no candidate jobs found; formal defaults remain frozen'
}

Write-ConfirmationLog "selection mode: $SelectionMode; confirmation steps: $ConfirmationSteps"

if ($DryRun) {
    $jobs |
        ForEach-Object {
            [pscustomobject]@{
                name = $_.Name
                variable = $_.Variable
                values = $_.Values
            }
        } |
        ConvertTo-Json -Depth 8
    return
}

$manifest = @()
foreach ($job in $jobs) {
    $outputDir = Join-Path $RootOutputDir $job.Name
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $stdout = Join-Path $outputDir 'worker.stdout.log'
    $stderr = Join-Path $outputDir 'worker.stderr.log'
    $args = @(
        'scripts/run_unpublished_sensitivity.py',
        '--variable', $job.Variable,
        '--values', $job.Values,
        '--backend', 'sionna',
        '--device', 'cuda',
        # The confirmation pass is an equal-budget three-seed experiment.
        # Keep the screening and confirmation arguments identical so seeds
        # 2026, 2027, and 2028 all receive the same optimizer-step budget.
        '--screening-steps', "$ConfirmationSteps",
        '--confirmation-steps', "$ConfirmationSteps",
        '--seeds', '2026,2027',
        '--confirmation-seed', '2028',
        '--validation-samples', '400',
        '--min-in-range-samples', '100',
        '--min-in-range-per-pilot', '20',
        '--batch-size', '8',
        '--microbatch-size', '4',
        '--generation-batch-size', '2',
        '--n-layers', '4',
        '--evaluation-batch-size', '2',
        '--output-dir', $outputDir,
        '--resume'
    )

    while ($true) {
        $summary = Read-Summary $outputDir
        if ($summary -and $summary.execution_complete -eq $true -and $summary.status -eq 'complete') {
            Write-ConfirmationLog "job $($job.Name) already complete; reusing it"
            break
        }
        Write-ConfirmationLog "starting job $($job.Name)"
        $worker = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        Wait-Process -Id $worker.Id
        $exitCode = $worker.ExitCode
        if ($null -eq $exitCode) {
            # Windows PowerShell can leave ExitCode unset after Wait-Process;
            # the summary is the authoritative completion signal.
            $exitCode = 0
        }
        Write-ConfirmationLog "job $($job.Name) exited with code $exitCode"
        $summary = Read-Summary $outputDir
        if ($exitCode -eq 0 -and $summary -and $summary.execution_complete -eq $true -and $summary.status -eq 'complete') {
            break
        }
        Write-ConfirmationLog "job $($job.Name) is incomplete; retrying with --resume"
        Start-Sleep -Seconds 10
    }
    $summary = Read-Summary $outputDir
    $manifest += [pscustomobject]@{
        name = $job.Name
        variable = $job.Variable
        values = $job.Values
        output_dir = $outputDir
        status = if ($summary) { $summary.status } else { 'missing' }
        accepted_record_count = if ($summary) { $summary.accepted_record_count } else { 0 }
    }
}

$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-ConfirmationLog "all confirmation jobs complete; manifest written to $manifestPath"
