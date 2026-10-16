param(
    [string]$OutputDirectory = 'outputs/reproduced_vla',
    [string]$PythonExecutable = '.venv-vla/Scripts/python.exe',
    [string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a new directory to preserve earlier experiments.' }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    function Invoke-Module {
        param([string]$ModuleName, [string[]]$ModuleArguments)
        & $PythonExecutable -m $ModuleName @ModuleArguments
        if ($LASTEXITCODE -ne 0) { throw "Module failed: $ModuleName" }
    }
    $datasetDirectory = Join-Path $OutputDirectory 'dataset'
    $manifestPath = Join-Path $OutputDirectory 'configurations.json'
    $modelDirectory = Join-Path $OutputDirectory 'model'
    $checkpointPath = Join-Path $modelDirectory 'policy.pt'
    $trainingReportPath = Join-Path $modelDirectory 'training.json'
    $freezeReceiptPath = Join-Path $OutputDirectory 'model_freeze.json'
    Invoke-Module 'ring_learning.vla_data' @('--output',$datasetDirectory)
    Invoke-Module 'ring_learning.vla_train' @('--dataset',$datasetDirectory,'--out',$modelDirectory,
        '--device',$Device,'--epochs','240','--batch-size','8','--decoder-steps','2000')
    Invoke-Module 'ring_learning.vla_evaluate' @('prepare','--dataset-manifest',(Join-Path $datasetDirectory 'manifest.json'),'--out',$manifestPath)
    Invoke-Module 'ring_learning.vla_evaluate' @('freeze','--manifest',$manifestPath,'--checkpoint',$checkpointPath,'--training-report',$trainingReportPath,'--out',$freezeReceiptPath)
    Invoke-Module 'ring_learning.vla_evaluate' @('run','--manifest',$manifestPath,'--receipt',$freezeReceiptPath,
        '--out',(Join-Path $OutputDirectory 'evaluation.json'),'--device','cpu')
    Invoke-Module 'ring_learning.vla_demo' @('--manifest',$manifestPath,'--checkpoint',$checkpointPath,
        '--output',(Join-Path $OutputDirectory 'student_demo.mp4'))
}
finally { Pop-Location }
