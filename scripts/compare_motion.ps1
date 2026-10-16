param(
    [string]$OutputDirectory = 'outputs/motion_ablation_reproduction',
    [string]$PythonExecutable = '.venv/Scripts/python.exe',
    [switch]$Plots
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    function Invoke-Module {
        param([string]$ModuleName, [string[]]$ModuleArguments)
        & $PythonExecutable -m $ModuleName @ModuleArguments
        if ($LASTEXITCODE -ne 0) { throw "Module failed: $ModuleName" }
    }
    Invoke-Module 'ring_learning.motion_ablation' @('prepare','--out',$OutputDirectory)
    Invoke-Module 'ring_learning.motion_ablation' @('run','--manifest',
        (Join-Path $OutputDirectory 'manifest.json'),'--out',(Join-Path $OutputDirectory 'evaluation.json'))
    if ($Plots) {
        Invoke-Module 'ring_learning.motion_ablation_analysis' @('--report',
            (Join-Path $OutputDirectory 'evaluation.json'),'--output',(Join-Path $OutputDirectory 'analysis'))
    }
}
finally { Pop-Location }
