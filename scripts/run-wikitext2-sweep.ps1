Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location "C:\repos\swarm-forge"

$py = ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "No existe: $py" }

$ckpt = "C:\repos\swarm-forge\runs\wikitext2_exp4_lr2p5e5\checkpoints\checkpoint_best_cycle_0001_valloss_2.5970.pt"
if (!(Test-Path $ckpt)) { throw "No existe checkpoint: $ckpt" }

$runsDir = "runs"
$logsDir = "runs\logs"
New-Item -ItemType Directory -Force -Path $runsDir | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$experiments = @(
    @{ Name = "lr3e5_steps10"; LR = "3e-5"; Steps = 10 },
    @{ Name = "lr3e5_steps20"; LR = "3e-5"; Steps = 20 },
    @{ Name = "lr4e5_steps10"; LR = "4e-5"; Steps = 10 },
    @{ Name = "lr4e5_steps20"; LR = "4e-5"; Steps = 20 },
    @{ Name = "lr5e5_steps10"; LR = "5e-5"; Steps = 10 },
    @{ Name = "lr5e5_steps20"; LR = "5e-5"; Steps = 20 }
)

$results = New-Object System.Collections.Generic.List[object]

foreach ($exp in $experiments) {
    $name = $exp.Name
    $lr = $exp.LR
    $steps = [int]$exp.Steps

    $outputDir = Join-Path $runsDir ("wikitext2_sweep_" + $name)
    $logFile = Join-Path $logsDir ("wikitext2_sweep_" + $name + ".log")

    Write-Host ""
    Write-Host ("== RUN {0} ==" -f $name) -ForegroundColor Cyan

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    & $py -m swarm_forge.core `
      --dataset-name "wikitext2" `
      --resume $ckpt `
      --output-dir $outputDir `
      --data-dir "data\wikitext2" `
      --device "cuda" `
      --amp `
      --dtype "float16" `
      --batch-size 8 `
      --block-size 256 `
      --n-layer 8 `
      --n-head 8 `
      --n-embd 512 `
      --dropout 0.2 `
      --learning-rate $lr `
      --max-iters-per-cycle $steps `
      --eval-iters 20 `
      --max-cycles 1 `
      --cycle-seconds 900 `
      --approval-threshold 60 `
      --score-threshold 70 `
      --focused-search `
      --reduced-roles-mode `
      --patch-trial-train-steps 20 `
      2>&1 | Tee-Object -FilePath $logFile

    $ErrorActionPreference = $prev

    $summaryLine = Get-Content $logFile | Select-String -Pattern "Cycle 0001 summary |" -SimpleMatch | Select-Object -Last 1
    $resumeLine  = Get-Content $logFile | Select-String -Pattern "Resumed from checkpoint" -SimpleMatch | Select-Object -Last 1
    $bestLine    = Get-Content $logFile | Select-String -Pattern "Saved improved checkpoint:" -SimpleMatch | Select-Object -Last 1
    $stopLine    = Get-Content $logFile | Select-String -Pattern "Guarded stop |" -SimpleMatch | Select-Object -Last 1

    $valLoss = [double]::NaN
    $trainLoss = [double]::NaN
    $baselineVal = [double]::NaN

    if ($summaryLine) {
        $m1 = [regex]::Match($summaryLine.Line, 'train_loss=([0-9.]+)')
        if ($m1.Success) { $trainLoss = [double]$m1.Groups[1].Value }

        $m2 = [regex]::Match($summaryLine.Line, 'val_loss=([0-9.]+)')
        if ($m2.Success) { $valLoss = [double]$m2.Groups[1].Value }
    }

    $chunkLine = Get-Content $logFile | Select-String -Pattern "Guarded chunk |" -SimpleMatch | Select-Object -First 1
    if ($chunkLine) {
        $m3 = [regex]::Match($chunkLine.Line, 'baseline_val=([0-9.]+)')
        if ($m3.Success) { $baselineVal = [double]$m3.Groups[1].Value }
    }

    $results.Add([pscustomobject]@{
        name         = $name
        lr           = $lr
        steps        = $steps
        baseline_val = $baselineVal
        train_loss   = $trainLoss
        val_loss     = $valLoss
        guarded_stop = [bool]($null -ne $stopLine)
        improved_ckpt= [bool]($null -ne $bestLine)
        log_file     = $logFile
    })
}

$summaryCsv = Join-Path $logsDir "wikitext2_sweep_summary.csv"
$results | Sort-Object val_loss | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "== RANKING FINAL ==" -ForegroundColor Green
$results |
    Sort-Object val_loss |
    Format-Table name,lr,steps,baseline_val,train_loss,val_loss,guarded_stop,improved_ckpt -AutoSize |
    Out-Host

Write-Host ""
Write-Host ("CSV => {0}" -f $summaryCsv) -ForegroundColor Green