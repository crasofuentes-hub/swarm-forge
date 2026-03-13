# VERIFY_BEST_TINYSHAKESPEARE.ps1
# Verifies existence and metadata of the best known TinyShakespeare checkpoint.

Write-Host "== SWARM FORGE CHECKPOINT VERIFICATION ==" -ForegroundColor Cyan

$repo = "C:\repos\swarm-forge"
$checkpoint = "runs\gpu_focused_search_v3\checkpoints\checkpoint_best_cycle_0007_valloss_1.6318.pt"
$metrics = "runs\gpu_focused_search_v3\logs\metrics.jsonl"

Write-Host "`n== REPOSITORY ROOT ==" -ForegroundColor Cyan
Get-Location | Out-Host

Write-Host "`n== CHECKPOINT ==" -ForegroundColor Cyan
if (Test-Path $checkpoint) {
    Get-Item $checkpoint | Select-Object FullName,Length,LastWriteTime | Out-Host
} else {
    Write-Host "Checkpoint not found!" -ForegroundColor Red
}

Write-Host "`n== METRICS FILE ==" -ForegroundColor Cyan
if (Test-Path $metrics) {
    Get-Item $metrics | Select-Object FullName,Length,LastWriteTime | Out-Host
} else {
    Write-Host "Metrics file not found!" -ForegroundColor Red
}

Write-Host "`n== LAST METRICS ENTRIES ==" -ForegroundColor Cyan
if (Test-Path $metrics) {
    Get-Content $metrics -Tail 10 | Out-Host
}

Write-Host "`n== AVAILABLE CHECKPOINTS ==" -ForegroundColor Cyan
Get-ChildItem "runs\gpu_focused_search_v3\checkpoints" |
    Select-Object Name,Length,LastWriteTime | Out-Host

Write-Host "`nVerification complete." -ForegroundColor Green
