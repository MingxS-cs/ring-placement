param(
    [string]$OutputDirectory = 'outputs/reproduced_v3',
    [string]$PythonExecutable = '.venv/Scripts/python.exe'
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    if (Test-Path -LiteralPath $OutputDirectory) {
        throw 'Output directory already exists. Choose another destination to preserve earlier results.'
    }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null

    function Invoke-PythonModule {
        param([string]$ModuleName, [string[]]$ModuleArguments)
        & $PythonExecutable -m $ModuleName @ModuleArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Module failed: $ModuleName (exit $LASTEXITCODE)"
        }
    }

    $endpointFile = Join-Path $OutputDirectory 'endpoints.json'
    $motionFile = Join-Path $OutputDirectory 'motion.json'
    $manifestFile = Join-Path $OutputDirectory 'configurations.json'
    $modelDirectory = Join-Path $OutputDirectory 'model'
    $checkpointFile = Join-Path $modelDirectory 'policy.pt'

    Invoke-PythonModule -ModuleName 'ring_learning.extract' -ModuleArguments @(
        '--output', $endpointFile, '--output-dir', (Join-Path $OutputDirectory 'endpoints'),
        '--visual-review', 'outputs/extraction/visual_review.json')
    Invoke-PythonModule -ModuleName 'ring_learning.motion' -ModuleArguments @(
        '--episodes', $endpointFile, '--output', $motionFile,
        '--output-dir', (Join-Path $OutputDirectory 'motion'),
        '--visual-review', 'outputs/motion/visual_review.json')
    # Freeze scenes before any model training; preserve the delivered scene seed.
    Invoke-PythonModule -ModuleName 'ring_learning.video_evaluate' -ModuleArguments @(
        '--prepare', '--real', $endpointFile, '--manifest', $manifestFile, '--seed', '20261017')
    Invoke-PythonModule -ModuleName 'ring_learning.video_train' -ModuleArguments @(
        '--real', $endpointFile, '--motion', $motionFile, '--output', $modelDirectory)
    Invoke-PythonModule -ModuleName 'ring_learning.video_evaluate' -ModuleArguments @(
        '--manifest', $manifestFile, '--checkpoint', $checkpointFile,
        '--output', (Join-Path $OutputDirectory 'evaluation'))
    # Demo reads original motion only for the visual panel; regenerated model
    # bindings and frozen scene provenance are still verified before control.
    Invoke-PythonModule -ModuleName 'ring_learning.video_demo' -ModuleArguments @(
        '--manifest', $manifestFile, '--checkpoint', $checkpointFile,
        '--output', (Join-Path $OutputDirectory 'student_demo.mp4'))
}
finally {
    Pop-Location
}
