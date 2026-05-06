param(
    [string]$TargetDir = "F:\TCNGATRE_github_clean",
    [switch]$Force,
    [switch]$InitGit
)

$ErrorActionPreference = "Stop"

$SourceDir = Split-Path -Parent $PSScriptRoot

if (Test-Path -LiteralPath $TargetDir) {
    if (-not $Force) {
        throw "Target directory '$TargetDir' already exists. Re-run with -Force to recreate it."
    }
    Remove-Item -LiteralPath $TargetDir -Recurse -Force
}

New-Item -ItemType Directory -Path $TargetDir | Out-Null

$rootFiles = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "plot_all_model_metrics.py",
    "plot_tcngatre_gpsdata_threshold_clipped.py",
    "run_all_models_all_datasets.py",
    "summarize_all_model_results.py"
)

$rootDirs = @(
    "TCNGATRE",
    "USAD",
    "Recurrent_AE",
    "TranAD",
    "OmniAnomaly",
    "BeatGAN",
    "common",
    "ablation",
    "hparam",
    "scripts"
)

$excludeDirs = @(
    ".git",
    ".claude",
    "__pycache__",
    ".ipynb_checkpoints",
    "dataset",
    "batch_logs",
    "runs"
)

$excludeFiles = @(
    "*.pyc",
    "*.pyo",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.png"
)

foreach ($file in $rootFiles) {
    $src = Join-Path $SourceDir $file
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $TargetDir $file) -Force
    }
}

foreach ($dir in $rootDirs) {
    $src = Join-Path $SourceDir $dir
    if (-not (Test-Path -LiteralPath $src)) {
        continue
    }

    $dest = Join-Path $TargetDir $dir
    New-Item -ItemType Directory -Path $dest -Force | Out-Null

    $robocopyArgs = @(
        $src,
        $dest,
        "/E",
        "/R:2",
        "/W:2",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP"
    )

    if ($excludeDirs.Count -gt 0) {
        $robocopyArgs += "/XD"
        $robocopyArgs += $excludeDirs
    }

    if ($excludeFiles.Count -gt 0) {
        $robocopyArgs += "/XF"
        $robocopyArgs += $excludeFiles
    }

    & robocopy @robocopyArgs | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed while exporting '$dir' (exit code $LASTEXITCODE)."
    }
}

if ($InitGit) {
    Push-Location $TargetDir
    try {
        git init | Out-Null
        git branch -M main
    }
    finally {
        Pop-Location
    }
}

Write-Host "Clean repository export created at $TargetDir"
