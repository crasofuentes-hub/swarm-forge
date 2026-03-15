Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location "C:\repos\swarm-forge"

$py = ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "No existe: $py" }

$ckpt = "C:\repos\swarm-forge\runs\guarded_real_from_16094\checkpoints\checkpoint_best_cycle_0001_valloss_1.5932.pt"
if (!(Test-Path $ckpt)) { throw "No existe checkpoint: $ckpt" }

New-Item -ItemType Directory -Force -Path "runs\logs" | Out-Null
$logFile = "runs\logs\reproduce_tinyshakespeare_reference.log"

$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"

& $py -m swarm_forge.core `
  --resume $ckpt `
  --output-dir "runs\reproduce_tinyshakespeare_reference" `
  --data-dir "data\tinyshakespeare" `
  --device "cuda" `
  --amp `
  --dtype "float16" `
  --batch-size 12 `
  --block-size 256 `
  --n-layer 8 `
  --n-head 8 `
  --n-embd 512 `
  --dropout 0.2 `
  --learning-rate 1e-4 `
  --max-iters-per-cycle 60 `
  --eval-iters 40 `
  --max-cycles 1 `
  --cycle-seconds 900 `
  --approval-threshold 60 `
  --score-threshold 70 `
  --focused-search `
  --reduced-roles-mode `
  --patch-trial-train-steps 40 `
  2>&1 | Tee-Object -FilePath $logFile

$ErrorActionPreference = $prev

Write-Host "`n== LINEAS CLAVE ==" -ForegroundColor Cyan
Get-Content $logFile |
  Select-String -Pattern @(
    "Effective optimizer lr after resume",
    "Resumed from checkpoint",
    "Guarded chunk |",
    "Guarded stop |",
    "Saved improved checkpoint:",
    "Cycle 0001 summary |"
  ) -SimpleMatch |
  ForEach-Object { $_.Line } |
  Out-Host