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
        param([string]$Name, [string[]]$Arguments)
        & $PythonExecutable -m $Name @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Module failed: $Name" }
    }
    $dataset = Join-Path $OutputDirectory 'dataset'
    $manifest = Join-Path $OutputDirectory 'configurations.json'
    $modelDirectory = Join-Path $OutputDirectory 'model'
    $checkpoint = Join-Path $modelDirectory 'policy.pt'
    $history = Join-Path $modelDirectory 'training.json'
    $receipt = Join-Path $OutputDirectory 'model_freeze.json'
    Invoke-Module 'ring_learning.vla_data' @('--output',$dataset)
    Invoke-Module 'ring_learning.vla_train' @('--dataset',$dataset,'--out',$modelDirectory,
        '--device',$Device,'--epochs','240','--batch-size','8','--decoder-steps','2000')
    Invoke-Module 'ring_learning.vla_evaluate' @('prepare','--dataset-manifest',(Join-Path $dataset 'manifest.json'),'--out',$manifest)
    Invoke-Module 'ring_learning.vla_evaluate' @('freeze','--manifest',$manifest,'--checkpoint',$checkpoint,'--training-report',$history,'--out',$receipt)
    Invoke-Module 'ring_learning.vla_evaluate' @('run','--manifest',$manifest,'--receipt',$receipt,
        '--out',(Join-Path $OutputDirectory 'evaluation.json'),'--device','cpu')
    Invoke-Module 'ring_learning.vla_demo' @('--manifest',$manifest,'--checkpoint',$checkpoint,
        '--output',(Join-Path $OutputDirectory 'student_demo.mp4'))
}
finally { Pop-Location }
