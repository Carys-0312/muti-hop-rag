<#
.SYNOPSIS
    打包项目用于上传到服务器（排除数据、密钥、虚拟环境、Git 历史）

.DESCRIPTION
    打包后目录里【不包含】：
      · data/raw、data/processed  → 索引源数据（3.7MB，服务器上重新生成）
      · data/eval/*.jsonl         → 评测集（服务器上可重建）
      · .env                      → ⚠️ 密钥，绝不上传
      · .venv                     → 本地虚拟环境
      · .git                      → Git 历史（部署不需要）
      · __pycache__              → Python 缓存
      · results/*.records.jsonl   → 逐条实验记录（500KB，部署不需要）

.EXAMPLE
    .\scripts\pack-for-deploy.ps1
    .\scripts\pack-for-deploy.ps1 -Output D:\mhrag.tar.gz
#>
param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

# 项目根 = 本脚本所在目录的上一级
$projectRoot = Split-Path -Parent $PSScriptRoot
$parentDir = Split-Path -Parent $projectRoot
$projName = Split-Path -Leaf $projectRoot

if (-not $Output) {
    $Output = Join-Path $parentDir "mhrag-deploy.tar.gz"
}

Write-Host ""
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " 打包项目用于部署" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " 项目目录 : $projectRoot"
Write-Host " 输出文件 : $Output"
Write-Host ""

# ── 安全检查：确认 .env 不会被包含 ──
$envFile = Join-Path $projectRoot ".env"
if (Test-Path $envFile) {
    Write-Host " ⚠️  检测到 .env（含 API Key）—— 打包时会【排除】它" -ForegroundColor Yellow
    Write-Host "     服务器上需要单独创建新的 .env" -ForegroundColor Yellow
}

if (Test-Path $Output) {
    Remove-Item $Output -Force
    Write-Host " 已删除旧的打包文件" -ForegroundColor DarkGray
}

# ── 打包（在父目录执行，这样压缩包里是 multi-hop-rag/ 一层目录）──
Push-Location $parentDir
try {
    Write-Host "`n 正在打包..." -ForegroundColor Cyan

    $tarArgs = @(
        "-czf", $Output,
        "--exclude=$projName/data/raw",
        "--exclude=$projName/data/processed",
        "--exclude=$projName/data/eval/*.jsonl",
        "--exclude=$projName/.env",
        "--exclude=$projName/.venv",
        "--exclude=$projName/.git",
        "--exclude=$projName/__pycache__",
        "--exclude=$projName/*/__pycache__",
        "--exclude=$projName/*/*/__pycache__",
        "--exclude=$projName/results/*.records.jsonl",
        "--exclude=$projName/results/*.metrics.json",
        "--exclude=$projName/*.egg-info",
        "--exclude=$projName/*/*.egg-info",
        "--exclude=$projName/.ruff_cache",
        "--exclude=$projName/.pytest_cache",
        $projName
    )

    & tar @tarArgs
    if ($LASTEXITCODE -ne 0) { throw "tar 打包失败（exit code $LASTEXITCODE）" }
}
finally {
    Pop-Location
}

# ── 结果 ──
$size = (Get-Item $Output).Length
Write-Host ""
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Green
Write-Host (" ✅ 打包完成：{0:N0} 字节 ({1:N1} KB)" -f $size, ($size / 1KB)) -ForegroundColor Green
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Green

# ── 列出压缩包内容（前 40 项，用于确认没打进敏感文件）──
Write-Host "`n 压缩包内容（前 40 项）:" -ForegroundColor Cyan
& tar -tzf $Output | Select-Object -First 40 | ForEach-Object { "   $_" }

Write-Host "`n ⚠️  请在上面的列表里确认：没有 .env、没有 data/processed" -ForegroundColor Yellow

Write-Host "`n 下一步 —— 上传到服务器：" -ForegroundColor Cyan
Write-Host "   scp `"$Output`" 用户名@服务器IP:/home/用户名/" -ForegroundColor White
Write-Host ""
