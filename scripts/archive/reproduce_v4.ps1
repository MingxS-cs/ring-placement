param(
    [string]$OutputDirectory = 'outputs/reproduced_v4',
    [string]$PythonExecutable = '.venv/Scripts/python.exe'
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location -LiteralPath $projectDirectory
try {
    if (Test-Path -LiteralPath $OutputDirectory) {
        throw 'Choose a new output directory to preserve earlier results and frozen receipts.'
    }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $dataDirectory = Join-Path $OutputDirectory 'data'
    $sourceDirectory = Join-Path $dataDirectory 'source'
    New-Item -ItemType Directory -Path $sourceDirectory -Force | Out-Null
    Copy-Item -LiteralPath 'data/source/359977334.mp4' -Destination $sourceDirectory

    function Invoke-PythonModule {
        param([string]$ModuleName, [string[]]$ModuleArguments)
        & $PythonExecutable -m $ModuleName @ModuleArguments
        if ($LASTEXITCODE -ne 0) { throw "Module failed: $ModuleName (exit $LASTEXITCODE)" }
    }

    $endpointFile = Join-Path $dataDirectory 'real_episodes.json'
    $motionFile = Join-Path $dataDirectory 'real_motion.json'
    $manifestFile = Join-Path $OutputDirectory 'configurations.json'
    $modelDirectory = Join-Path $OutputDirectory 'model'
    $checkpointFile = Join-Path $modelDirectory 'policy.json'
    $blindReceipt = Join-Path $OutputDirectory 'reproduction_blind_receipt.json'
    $modelReceipt = Join-Path $OutputDirectory 'model_freeze.json'

    Invoke-PythonModule -ModuleName 'ring_learning.extract' -ModuleArguments @(
        '--output', $endpointFile, '--output-dir', (Join-Path $OutputDirectory 'endpoints'),
        '--visual-review', 'outputs/extraction/visual_review.json')
    Invoke-PythonModule -ModuleName 'ring_learning.motion' -ModuleArguments @(
        '--episodes', $endpointFile, '--output', $motionFile,
        '--output-dir', (Join-Path $OutputDirectory 'motion'),
        '--visual-review', 'outputs/motion/visual_review.json')
    # Recheck every regenerated label value before rebinding metadata-only hashes.
    & $PythonExecutable 'scripts/archive/reproduction_receipt.py' --real $endpointFile --motion $motionFile --output $blindReceipt
    if ($LASTEXITCODE -ne 0) { throw 'Regenerated measurements did not match the delivered source labels.' }
    Invoke-PythonModule -ModuleName 'ring_learning.generalization_evaluate' -ModuleArguments @(
        '--prepare', '--real', $endpointFile, '--manifest', $manifestFile)
    Invoke-PythonModule -ModuleName 'ring_learning.generalization_train' -ModuleArguments @(
        '--real', $endpointFile, '--motion', $motionFile, '--output', $modelDirectory)
    Invoke-PythonModule -ModuleName 'ring_learning.generalization_evaluate' -ModuleArguments @(
        '--freeze-model', '--manifest', $manifestFile, '--checkpoint', $checkpointFile,
        '--blind-receipt', $blindReceipt, '--receipt', $modelReceipt)
    Invoke-PythonModule -ModuleName 'ring_learning.generalization_evaluate' -ModuleArguments @(
        '--manifest', $manifestFile, '--checkpoint', $checkpointFile, '--receipt', $modelReceipt,
        '--output', (Join-Path $OutputDirectory 'evaluation'))
    Invoke-PythonModule -ModuleName 'ring_learning.generalization_fidelity' -ModuleArguments @(
        '--checkpoint', $checkpointFile, '--model-freeze-receipt', $modelReceipt,
        '--output', (Join-Path $OutputDirectory 'fidelity'))
    Invoke-PythonModule -ModuleName 'ring_learning.generalization_demo' -ModuleArguments @(
        '--manifest', $manifestFile, '--checkpoint', $checkpointFile,
        '--output', (Join-Path $OutputDirectory 'student_demo.mp4'))
}
finally { Pop-Location }
