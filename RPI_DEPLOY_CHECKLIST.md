# Raspberry Pi Deploy Checklist (AFLFantasy Predictions)

Serves the AFL Predictions web app at `aflfantasy.delophant.xyz` (port 8001 behind nginx).
Training and scraping stay on your PC — the Pi only runs `python -m web`.

## Layout

| Purpose | Path |
|---------|------|
| App checkout | `/srv/aflfantasy/prod` |
| Predictions data | `/srv/aflfantasy/prod/web/data/predictions/` |
| systemd unit | `/etc/systemd/system/aflfantasy-prod.service` |
| nginx vhost | `/etc/nginx/sites-available/aflfantasy-prod.conf` |

## 1) One-time Pi setup

**Prerequisite:** `scripts/`, `deploy/`, and `requirements-rpi.txt` must be on GitHub before the steps below work. If a shallow clone only shows `src/`, `README.md`, etc. (no `scripts/`), push the deploy scaffolding from your PC first, or use the manual layout fallback below.

Clone the **`web_predict`** branch (has the `web/` app). Default `main` does not.

```bash
sudo apt update
sudo apt install -y python3 python3-venv nginx git
```

### When to run `setup-rpi-layout.sh`

**Once**, the first time you deploy AFLFantasy on the Pi. It only creates `/srv/aflfantasy/prod` and sets ownership — safe to re-run anytime.

**Where to run it from:** any directory, as long as you have a copy of this repo with `scripts/setup-rpi-layout.sh`. Typical first-time flow:

```bash
# Option A (recommended): shallow clone to home, run layout, then full prod checkout
cd ~
git clone --depth 1 -b web_predict git@github.com:Delophant/AFLFantasy-DNN.git AFLFantasy-DNN-setup
cd ~/AFLFantasy-DNN-setup
chmod +x scripts/setup-rpi-layout.sh
./scripts/setup-rpi-layout.sh

cd /srv/aflfantasy
git clone -b web_predict git@github.com:Delophant/AFLFantasy-DNN.git prod
cd prod
python3 -m venv .venv
.venv/bin/pip install -r requirements-rpi.txt
```

The home clone is throwaway — only needed so you can run the layout script before `/srv/aflfantasy/prod` exists. The prod clone under `/srv/aflfantasy/prod` is the real install.

If you already cloned into `/srv/aflfantasy/prod` first, run layout from there instead:

```bash
cd /srv/aflfantasy/prod
chmod +x scripts/setup-rpi-layout.sh
./scripts/setup-rpi-layout.sh
```

**Manual layout fallback** (if `scripts/setup-rpi-layout.sh` is not in the clone yet):

```bash
sudo mkdir -p /srv/aflfantasy/prod
sudo chown -R delophant:delophant /srv/aflfantasy
```

You do **not** run this script from your PC — only on the Pi (SSH session as `delophant`).

Copy systemd + nginx templates:

```bash
sudo cp deploy/aflfantasy-prod.service /etc/systemd/system/
sudo cp deploy/aflfantasy-nginx.conf /etc/nginx/sites-available/aflfantasy-prod.conf
sudo ln -sf /etc/nginx/sites-available/aflfantasy-prod.conf /etc/nginx/sites-enabled/aflfantasy-prod.conf
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl enable --now aflfantasy-prod
```

**Important:** Only ever `pip install -r requirements-rpi.txt` on the Pi — never `requirements.txt` (TensorFlow).

## 2) Weekly workflow (from your PC)

```powershell
# 1. Refresh stats (if needed)
.\.venv\Scripts\python.exe src\scrape_footywire.py 2021 2026

# 2. Predict next round (halts if already done)
.\.venv\Scripts\python.exe src\predict_upcoming.py

# 3. Push CSVs + manifest to Pi
.\scripts\publish-predictions.ps1
```

Or predict and publish in one step:

```powershell
.\.venv\Scripts\python.exe src\predict_upcoming.py --deploy
```

## 3) Web code updates (from your PC)

After pushing to GitHub:

```powershell
.\scripts\deploy-rpi.ps1
```

This SSHs to the Pi, runs `git pull`, reinstalls `requirements-rpi.txt` if needed, and restarts the service.

## 4) Verify

```bash
curl -s http://127.0.0.1:8001/ | head
systemctl status aflfantasy-prod
```

LAN: `http://192.168.1.13/` won't show this app unless you use the `Host: aflfantasy.delophant.xyz` header — same pattern as WheelofFeed.
