# Publish prediction CSVs + manifest to the Raspberry Pi.
#
# Usage:
#   .\scripts\publish-predictions.ps1
#   .\scripts\publish-predictions.ps1 -DryRun
#
# Requires: OpenSSH (ssh/scp), key-based login to the Pi.

param(
    [string]$PiHost = "192.168.1.13",
    [string]$PiUser = "delophant",
    [string]$RemoteDir = "/srv/aflfantasy/prod/web/data/predictions",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LocalDir = Join-Path $RepoRoot "web\data\predictions"
$Remote = "${PiUser}@${PiHost}"

if (-not (Test-Path $LocalDir)) {
    throw "Missing predictions folder: $LocalDir"
}

$files = @(
    Get-ChildItem -LiteralPath $LocalDir -Filter "*.csv" -File
    Get-ChildItem -LiteralPath $LocalDir -Filter "manifest.json" -File -ErrorAction SilentlyContinue
) | Where-Object { $_ }

if ($files.Count -eq 0) {
    throw "No CSV or manifest.json files in $LocalDir"
}

Write-Host "==> Publish predictions ($($files.Count) files) -> ${Remote}:${RemoteDir}"

if ($DryRun) {
    $files | ForEach-Object { Write-Host "    $($_.Name)" }
    exit 0
}

ssh $Remote "mkdir -p '${RemoteDir}'"
$paths = @($files | ForEach-Object { $_.FullName })
& scp @paths "${Remote}:${RemoteDir}/"

Write-Host "Publish complete."
