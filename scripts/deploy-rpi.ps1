# Deploy AFLFantasy web app to Raspberry Pi (git pull + restart).
#
# Usage:
#   .\scripts\deploy-rpi.ps1
#   .\scripts\deploy-rpi.ps1 -DryRun
#
# First-time setup on the Pi — see RPI_DEPLOY_CHECKLIST.md

param(
    [string]$PiHost = "192.168.1.13",
    [string]$PiUser = "delophant",
    [string]$AppDir = "/srv/aflfantasy/prod",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Remote = "${PiUser}@${PiHost}"

# Pipe to bash -s so Windows CRLF in this file cannot break remote commands.
$cmd = @"
set -e
cd '${AppDir}'
# Predictions are synced via publish-predictions.ps1 (scp), not git — discard local diffs before pull.
git checkout -- web/data/predictions/ 2>/dev/null || true
git pull
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -r requirements-rpi.txt
sudo systemctl restart aflfantasy-prod
echo 'Deploy complete.'
"@.Replace("`r`n", "`n").Replace("`r", "`n")

Write-Host "==> Deploy web app on ${Remote}:${AppDir}"

if ($DryRun) {
    Write-Host $cmd
    exit 0
}

$cmd | ssh $Remote "bash -s"
