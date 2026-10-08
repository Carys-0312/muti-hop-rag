<#
.SYNOPSIS
    清理 Docker 镜像：只保留「指定 digest 的镜像」+「本项目需要的镜像」。

.DESCRIPTION
    默认是【干跑模式】——只打印计划，不删任何东西。
    确认清单没问题后，加 -Execute 才真正删除。

.EXAMPLE
    # ① 先干跑，看看会删哪些
    .\scripts\cleanup-images.ps1

    # ② 确认无误后真正执行
    .\scripts\cleanup-images.ps1 -Execute
#>
param(
    # 要额外保留的镜像 digest（从 docker images --digests 里拿到的）
    [string]$KeepDigest = "858f009f9709ce576febc734aa78b8f6d624b82571f9ddb6bda4377c833b3499",

    # 真正执行删除（不加就是干跑）
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

# ── 本项目需要的镜像（按仓库名前缀匹配）──────────────────────
$keepPrefixes = @(
    "redis",            # compose 用 redis:7-alpine
    "python",           # Dockerfile 的 FROM python:3.12-slim
    "qdrant/qdrant",    # compose 的向量库
    "multi-hop-rag"     # 构建出来的应用镜像
)

Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Docker 镜像清理" -ForegroundColor Cyan
Write-Host " 模式: $(if ($Execute) { '⚠️  真正执行删除' } else { '✅ 干跑（不会删任何东西）' })" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Cyan

# ── 读取镜像列表 ──────────────────────────────────────────
$raw = docker images --no-trunc --format "{{.ID}}|{{.Repository}}|{{.Tag}}|{{.Size}}"
if (-not $raw) {
    Write-Host "`n没有找到任何镜像，无需清理。" -ForegroundColor Yellow
    exit 0
}

$keep = @()
$drop = @()
$digestMatched = $false

foreach ($line in $raw) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line -split '\|'
    if ($parts.Count -lt 4) { continue }

    $id = $parts[0]; $repo = $parts[1]; $tag = $parts[2]; $size = $parts[3]

    $reason = $null

    # ① 匹配指定 digest
    if ($id -like "*$KeepDigest*") {
        $reason = "按 digest 指定保留"
        $digestMatched = $true
    }
    # ② 匹配项目需要的镜像
    else {
        foreach ($p in $keepPrefixes) {
            if ($repo -like "$p*") { $reason = "项目需要"; break }
        }
    }

    $obj = [pscustomobject]@{
        Id = $id; Repo = $repo; Tag = $tag; Size = $size; Reason = $reason
    }
    if ($reason) { $keep += $obj } else { $drop += $obj }
}

# ── 打印"保留"清单 ───────────────────────────────────────
Write-Host "`n【保留】共 $($keep.Count) 个" -ForegroundColor Green
if ($keep.Count -eq 0) {
    Write-Host "  （无）" -ForegroundColor DarkGray
} else {
    $keep | ForEach-Object {
        $short = $_.Id.Substring(0, [Math]::Min(12, $_.Id.Length))
        Write-Host ("  {0,-12}  {1,-28} {2,-12} {3,-10} {4}" -f $short, $_.Repo, $_.Tag, $_.Size, $_.Reason)
    }
}

# ── 打印"删除"清单 ───────────────────────────────────────
Write-Host "`n【将删除】共 $($drop.Count) 个" -ForegroundColor Yellow
if ($drop.Count -eq 0) {
    Write-Host "  （无）" -ForegroundColor DarkGray
} else {
    $drop | ForEach-Object {
        $short = $_.Id.Substring(0, [Math]::Min(12, $_.Id.Length))
        Write-Host ("  {0,-12}  {1,-28} {2,-12} {3}" -f $short, $_.Repo, $_.Tag, $_.Size)
    }
}

# ── 提示 digest 匹配情况 ─────────────────────────────────
if (-not $digestMatched) {
    Write-Host "`n⚠️  没有镜像匹配你给的 digest: $KeepDigest" -ForegroundColor Yellow
    Write-Host "    可能是：① 它是个 RepoDigest 不是 Image ID → 试试先跑：" -ForegroundColor Yellow
    Write-Host "       docker images --digests --no-trunc" -ForegroundColor DarkGray
    Write-Host "    ② 或者镜像已经被删了" -ForegroundColor Yellow
}

# ── 干跑就到此为止 ───────────────────────────────────────
if (-not $Execute) {
    Write-Host "`n──────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host " 这是【干跑】，什么都没删。" -ForegroundColor Cyan
    Write-Host " 确认上面的清单没问题后，执行：" -ForegroundColor Cyan
    Write-Host "     .\scripts\cleanup-images.ps1 -Execute" -ForegroundColor White
    Write-Host "──────────────────────────────────────────────────────" -ForegroundColor DarkGray
    exit 0
}

# ── 真正执行 ─────────────────────────────────────────────
if ($drop.Count -eq 0) {
    Write-Host "`n没有需要删除的镜像。" -ForegroundColor Green
    exit 0
}

Write-Host "`n开始删除..." -ForegroundColor Yellow
$ok = 0; $fail = 0
foreach ($img in $drop) {
    $ref = "$($img.Repo):$($img.Tag)"
    if ($img.Repo -eq "<none>") { $ref = $img.Id }   # 悬空镜像用 ID
    $out = docker rmi $ref 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ $ref" -ForegroundColor Green
        $ok++
    } else {
        Write-Host "  ❌ $ref" -ForegroundColor Red
        Write-Host "     $out" -ForegroundColor DarkGray
        $fail++
    }
}

Write-Host "`n完成：成功 $ok 个，失败 $fail 个" -ForegroundColor Cyan
if ($fail -gt 0) {
    Write-Host "失败通常是「镜像正被容器占用」，先清理容器再试：" -ForegroundColor Yellow
    Write-Host "    docker container prune -f" -ForegroundColor DarkGray
}
