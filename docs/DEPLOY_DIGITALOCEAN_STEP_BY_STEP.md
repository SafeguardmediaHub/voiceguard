# VoiceGuard — DigitalOcean Deployment, Click by Click

A complete walkthrough from an empty DigitalOcean account to a running, verified,
VPC-private VoiceGuard API. Every browser click and every terminal command.

**This is the beginner-friendly companion to [`DEPLOYMENT.md`](../DEPLOYMENT.md).**
That file is the terse operator runbook using `doctl`. This file does the same
thing through the web control panel. Same end state — pick whichever you prefer.
Where they disagree, `DEPLOYMENT.md` wins on the application details (`.env`,
model bundle, drift), because it is the file the tests fence.

---

## Legend

Every step is tagged with **where** you do it:

| Tag | Meaning |
|---|---|
| 🖱️ **PANEL** | In your web browser, at `cloud.digitalocean.com` |
| 💻 **WORKSTATION** | A terminal on your own Windows machine (PowerShell or Git Bash) |
| 🖥️ **VG DROPLET** | A terminal SSH'd into the VoiceGuard droplet |
| 🖥️ **BACKEND** | A terminal SSH'd into your *backend* droplet (the thing that calls VoiceGuard) |

Running a command on the wrong machine is the single most common way this goes
wrong. Check the tag before you paste.

> **On UI drift:** DigitalOcean renames and moves things every few months. Where a
> button label may have changed, I give the **direct URL** as well — those have
> been stable for years. If a label does not match, use the URL and look for the
> nearest equivalent.

---

## Table of contents

- [Phase 0 — Before you start](#phase-0--before-you-start)
- [Phase 1 — Account, team, project](#phase-1--account-team-project)
- [Phase 2 — Pick your region and stick to it](#phase-2--pick-your-region-and-stick-to-it)
- [Phase 3 — Create the VPC](#phase-3--create-the-vpc)
- [Phase 4 — Create the Spaces bucket](#phase-4--create-the-spaces-bucket)
- [Phase 5 — Create Spaces access keys](#phase-5--create-spaces-access-keys)
- [Phase 6 — Create the Container Registry](#phase-6--create-the-container-registry)
- [Phase 7 — Create an API token](#phase-7--create-an-api-token)
- [Phase 8 — Add your SSH key](#phase-8--add-your-ssh-key)
- [Phase 9 — Create the droplet](#phase-9--create-the-droplet)
- [Phase 10 — Find the private IP](#phase-10--find-the-private-ip)
- [Phase 11 — Create the Cloud Firewall](#phase-11--create-the-cloud-firewall)
- [Phase 12 — Build and push the image](#phase-12--build-and-push-the-image)
- [Phase 13 — Configure and start the stack](#phase-13--configure-and-start-the-stack)
- [Phase 14 — Issue the backend an API key](#phase-14--issue-the-backend-an-api-key)
- [Phase 15 — Wire up the backend](#phase-15--wire-up-the-backend)
- [Phase 16 — Nightly encrypted backups](#phase-16--nightly-encrypted-backups)
- [Phase 17 — Drift monitoring](#phase-17--drift-monitoring)
- [Phase 18 — Monitoring and alerts](#phase-18--monitoring-and-alerts)
- [Phase 19 — Go-live checklist](#phase-19--go-live-checklist)
- [Day-2 operations](#day-2-operations)
- [Troubleshooting](#troubleshooting)
- [Appendix A — What it costs](#appendix-a--what-it-costs)
- [Appendix B — Every value you need to write down](#appendix-b--every-value-you-need-to-write-down)

---

## Phase 0 — Before you start

### 0.1 What you are building

```
                       Public Internet
                             │  HTTPS
                    ┌────────▼─────────┐
                    │ Backend Droplet  │   your app — the only public surface
                    └────────┬─────────┘
                             │  HTTPS over the private VPC, Bearer API key
        ┌────────────────────▼──────────────────────────┐
        │  VoiceGuard Droplet — 4 vCPU / 8 GB, CPU only │
        │  Caddy binds the VPC IP only, never 0.0.0.0   │
        │                                               │
        │    caddy :8443 ──► api (gunicorn, 3 workers)  │
        │                      │  SQLite queue on the   │
        │                      ▼  vg-data volume        │
        │                    worker  × 1                │
        └───────────────────────────────────────────────┘
                             ▲ docker pull
                    ┌────────┴─────────┐
                    │ Container Reg.   │◄── you build and push this
                    └──────────────────┘
```

`api` and `worker` are the **same image** with different arguments. They talk only
through the shared `/data` volume — a SQLite job queue in WAL mode plus uploaded
files. That is why this is one droplet and not a cluster.

**VoiceGuard is never reachable from the public internet.** Only your backend
droplet, over the private VPC, on port 8443. That is the whole security model.

### 0.2 What you need before Phase 1

- [ ] A payment card (DigitalOcean bills monthly; expect roughly **$75–90/month** — see [Appendix A](#appendix-a--what-it-costs))
- [ ] This repository checked out on your workstation
- [ ] **Docker Desktop** installed and running on Windows — <https://www.docker.com/products/docker-desktop/>
- [ ] **Python 3.13** on your workstation (needed to run `bundle_registry.py`)
- [ ] **Git** with Git Bash (ships with Git for Windows)
- [ ] Roughly **20 GB free disk** on your workstation — the image is ~3 GB and the build cache is larger
- [ ] A decent upload connection. You will push a ~3 GB image once. On a slow uplink see [12.6](#126-if-your-upload-is-too-slow)

### 0.3 A word about the model weights

The model weights are **not in git** (`.gitignore` excludes `*.pt`). Your local
`model_store/` may be 1.2 GB across several bundles, but only the **active
bundle** (`v9h`, ~387 MB) goes into the image — `.dockerignore` keeps the rest out.

If your workstation does not already have `model_store/v9h/`, you will pull it
from Spaces in [Phase 12](#phase-12--build-and-push-the-image). That is the only
reason Spaces exists at build time.

---

## Phase 1 — Account, team, project

### 1.1 🖱️ PANEL — Sign up or log in

1. Open a browser and go to **<https://cloud.digitalocean.com>**
2. If you have no account, click **Sign Up** and complete registration — email
   verification plus a payment method. DigitalOcean will not let you create a
   droplet until a card is on file.
3. Once logged in you land on a page headed with a project name (a new account
   gets one called **first-project**).

### 1.2 🖱️ PANEL — Create a dedicated project

A project is just a folder for grouping resources, but it makes billing and
cleanup much easier later.

1. Look at the **far-left sidebar**. At the very top is your project list.
2. Scroll to the **bottom of the project list** and click **+ New Project**.
   (If you do not see it, click the **Create** button — the green one at the top
   right of the page — and choose **Projects** from the dropdown.)
3. Fill in the form:
   - **Name**: `voiceguard-prod`
   - **Description**: `VoiceGuard audio deepfake detector — production`
   - **Tell us what it's for**: choose **Service or API**
4. Click **Create Project**.
5. It will ask *"Want to move existing resources?"* — click **Skip for now**.

Every resource you create from here on, assign it to **voiceguard-prod**. Most
create-forms have a **Select Project** dropdown near the bottom.

---

## Phase 2 — Pick your region and stick to it

This is a decision, not a click, and getting it wrong costs you a rebuild.

**The rule:** your VoiceGuard droplet, your backend droplet, and your VPC **must
all be in the same region**. A VPC does not span regions, and without a shared VPC
your backend cannot reach VoiceGuard privately — which breaks the entire security
model.

| Region slug | Location | Use if |
|---|---|---|
| `fra1` | Frankfurt | European users; this is what the repo docs use as the example |
| `lon1` | London | UK data-residency requirements |
| `nyc3` | New York | North American users |
| `ams3` | Amsterdam | EU alternative to Frankfurt |
| `sgp1` | Singapore | Asia-Pacific |
| `blr1` | Bangalore | India |

> **If your backend droplet already exists**, you do not get to choose. Use
> whatever region *it* is in. Go look it up now: sidebar **Droplets**, click the
> backend droplet, and read the region under its name.

**Write your choice down.** Everywhere below that says `<REGION>`, substitute it.
This document uses `fra1` in examples.

Not every region offers Spaces. As of now Spaces is available in `nyc3`, `ams3`,
`sfo3`, `sgp1`, `fra1`, `syd1`, `blr1`. If your compute region has no Spaces, that
is fine — put Spaces in the nearest region that has it, since it is only used at
build time and for backups, not in the request path.

---

## Phase 3 — Create the VPC

The VPC is the private network your two droplets share. Create it **before** the
droplet, because you can only attach a droplet to a VPC at creation time.

### 3.1 🖱️ PANEL — Check whether you already have one

1. In the left sidebar, find the **NETWORKING** section and click **Networking**.
   Direct URL: **<https://cloud.digitalocean.com/networking/vpc>**
2. Along the top of the page is a row of tabs: *Load Balancers · Domains · VPC ·
   Firewalls · Reserved IPs · PTR Records*. Click the **VPC** tab.
   (In some accounts this reads **VPC Network**.)
3. You will see a table. DigitalOcean auto-creates a **default VPC per region**.

**If a VPC already exists in your region and your backend droplet is on it — use
that one. Skip to [3.3](#33--panel--record-the-ip-range).** Creating a second VPC
in the same region and putting VoiceGuard on it would isolate it from your backend.

### 3.2 🖱️ PANEL — Create one (only if needed)

1. On the **VPC** tab, click the **Create VPC Network** button (top right).
2. **Choose a datacenter region** — click your `<REGION>` tile.
3. **Network name**: `voiceguard-vpc`
4. **Description**: `Private network for VoiceGuard and backend`
5. **IP range**: leave it on **Generate an IP range for me**. Only override this
   if you are peering with something else and need a specific CIDR.
6. **Select Project**: `voiceguard-prod`
7. Click **Create VPC Network**.

### 3.3 🖱️ PANEL — Record the IP range

Click into your VPC. Note the **IP Range** — something like `10.114.0.0/20`. Every
droplet on this VPC gets a private address inside that range. You will need those
addresses in [Phase 11](#phase-11--create-the-cloud-firewall).

📝 **Write down:** `VPC name`, `VPC IP range`, `region`.

---

## Phase 4 — Create the Spaces bucket

Spaces is DigitalOcean's S3-compatible object storage. VoiceGuard uses it for two
things, **neither of which is in the request path**:

1. **Build time** — storing model bundles so any machine can rebuild the image
2. **Nightly backups** — encrypted copies of the job DB, API keys, and audit log

Production itself holds no Spaces credentials. The model is baked into the image.

### 4.1 🖱️ PANEL — Create the bucket

1. In the left sidebar under **MANAGE**, click **Spaces Object Storage**.
   Direct URL: **<https://cloud.digitalocean.com/spaces>**
2. Click **Create a Spaces Bucket** (or the green **Create** button → **Spaces
   Object Storage**).
3. **Choose a datacenter region** — pick `<REGION>`, or the nearest Spaces-capable
   region.
4. **Enable CDN**: leave this **OFF**. This bucket is private; a CDN is pointless
   and costs money.
5. **Choose a unique name**: bucket names are globally unique across all of
   DigitalOcean, so `voiceguard` is almost certainly taken. Use something like
   `voiceguard-models-<yourcompany>`. Lowercase letters, numbers and hyphens only.
6. **Select Project**: `voiceguard-prod`
7. Click **Create a Spaces Bucket**.

### 4.2 🖱️ PANEL — Lock it down

1. You are now inside the bucket. Click the **Settings** tab.
2. Find **File Listing** and confirm it is set to **Restricted**. If it says
   *Public*, click **Edit** and change it. Model weights must not be world-readable.

### 4.3 🖱️ PANEL — Consider a second bucket for backups

`DEPLOYMENT.md` §9.1 uses a separate bucket for backups. That separation is worth
having: the backup credentials live on the production droplet, and you do not want
those same credentials able to overwrite your model bundles.

Repeat 4.1 with the name `voiceguard-backups-<yourcompany>`.

📝 **Write down:** `SPACES_BUCKET` (models), `SPACES_BUCKET` (backups), and the
**endpoint**, which is `https://<REGION>.digitaloceanspaces.com` — e.g.
`https://fra1.digitaloceanspaces.com`.

---

## Phase 5 — Create Spaces access keys

### 5.1 🖱️ PANEL — Generate the key pair

1. Scroll to the **bottom of the left sidebar**. Under **SETTINGS**, click **API**.
   Direct URL: **<https://cloud.digitalocean.com/account/api/spaces>**
2. Along the top are tabs: *Tokens · OAuth Applications · Spaces Keys*. Click
   **Spaces Keys**.
3. Click **Generate New Key**.
4. **Name**: `voiceguard-build` (this one is for your workstation)
5. If your account offers a scope selector, choose the models bucket with
   **Read/Write**. Older accounts issue account-wide keys with no scoping.
6. Click **Create Access Key**.

### 5.2 ⚠️ Copy the secret NOW

The panel shows two strings:

- **Access Key** (~20 chars) — visible forever, this is `SPACES_KEY`
- **Secret Key** (~43 chars) — **shown exactly once**, this is `SPACES_SECRET`

Copy the secret into your password manager before you navigate away. If you lose
it, your only option is to delete the key and generate a new one.

### 5.3 🖱️ PANEL — Generate a second key for backups

Repeat 5.1–5.2 with the name `voiceguard-backup`, scoped to the **backups**
bucket. This key goes on the production droplet in
[Phase 16](#phase-16--nightly-encrypted-backups); the build key never does.

📝 **Write down:** two `SPACES_KEY` / `SPACES_SECRET` pairs, clearly labelled.

---

## Phase 6 — Create the Container Registry

### 6.1 ⚠️ Read this before you pick a plan

**The VoiceGuard image is about 3 GB uncompressed** (Python 3.13 + torch 2.12 +
transformers + the baked-in `facebook/wav2vec2-base` weights + the ~387 MB `v9h`
bundle). Registries store compressed layers, so expect roughly **1.2–1.5 GB per
tag** on disk.

Rollback works by *deploying a previous SHA tag*, so you must keep old tags around.
That makes the plan choice load-bearing:

| Plan | Storage | Repositories | Price | Verdict for VoiceGuard |
|---|---|---|---|---|
| **Starter** | 500 MiB | 1 | Free | ❌ **Will not work.** One layer of this image exceeds the whole quota. |
| **Basic** | 5 GiB | 5 | $5/mo | ⚠️ Holds about 3 tags. Workable but you will be pruning constantly. |
| **Professional** | 100 GiB | Unlimited | $20/mo | ✅ **Recommended.** Comfortable tag history for real rollbacks. |

Storage above the included quota is billed as overage, so Basic does not hard-fail
— it just quietly costs more while giving you less room.

### 6.2 🖱️ PANEL — Create it

1. In the left sidebar under **MANAGE**, click **Container Registry**.
   Direct URL: **<https://cloud.digitalocean.com/registry>**
2. Click **Create Container Registry** (or **Get Started**).
3. **Registry name**: this becomes part of every image path, so keep it short and
   permanent — e.g. `safeguardmedia`. It is globally unique and **cannot be
   renamed later**.
4. **Choose a datacenter region**: `<REGION>` — same as the droplet, so pulls are
   fast and free over the internal network.
5. **Choose a plan**: **Professional** (see 6.1).
6. Click **Create Registry**.

### 6.3 Note your image path

Your images will live at:

```
registry.digitalocean.com/<REGISTRY_NAME>/voiceguard:<TAG>
```

For example `registry.digitalocean.com/safeguardmedia/voiceguard:a1b2c3d`.

📝 **Write down:** `REGISTRY_NAME`.

---

## Phase 7 — Create an API token

You need this to log Docker into the registry, from both your workstation and the
droplet.

### 7.1 🖱️ PANEL — Generate

1. Left sidebar, bottom, under **SETTINGS** → **API**.
   Direct URL: **<https://cloud.digitalocean.com/account/api/tokens>**
2. Make sure you are on the **Tokens** tab (the first one).
3. Click **Generate New Token**.
4. **Token name**: `voiceguard-registry`
5. **Expiration**: choose **90 days** if you are willing to rotate, or **No
   expiry** if you would rather not have a deploy fail at 2am. Rotating is better
   practice; pick what you will actually maintain.
6. **Scopes**: choose **Custom Scopes** if offered, and grant **Read** and
   **Write** on **Container Registry** only. If your account only offers
   Read/Write account-wide, accept it — but treat the token as highly sensitive.
7. Click **Generate Token**.

### 7.2 ⚠️ Copy it now

The token (`dop_v1_...`) is shown **once**. Put it in your password manager.

📝 **Write down:** `DO_API_TOKEN`.

---

## Phase 8 — Add your SSH key

DigitalOcean can create a droplet with password auth, but do not. Use a key.

### 8.1 💻 WORKSTATION — Generate a key (skip if you have one)

Open **PowerShell** and run:

```powershell
ssh-keygen -t ed25519 -C "voiceguard-admin"
```

- When asked *"Enter file in which to save the key"*, press **Enter** to accept
  the default `C:\Users\<you>\.ssh\id_ed25519`.
- When asked for a passphrase, set one. It protects the key at rest.

Now print the **public** key:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Copy the entire line of output. It starts with `ssh-ed25519 AAAA...` and ends with
your comment. **Never** copy the file without `.pub` — that one is the private key
and must never leave your machine.

### 8.2 🖱️ PANEL — Register it

1. Left sidebar, bottom → **Settings** → **Security** tab.
   Direct URL: **<https://cloud.digitalocean.com/account/security>**
2. Scroll down to the **SSH keys** section.
3. Click **Add SSH Key**.
4. Paste the public key into the large **SSH key content** box.
5. **Name**: `michael-workstation` (or your machine name).
6. Click **Add SSH Key**.

---

## Phase 9 — Create the droplet

### 9.1 🖱️ PANEL — Open the create form

1. Click the green **Create** button at the top right of any page.
2. Choose **Droplets** from the dropdown.
   Direct URL: **<https://cloud.digitalocean.com/droplets/new>**

Now work down the form. It is long — take it section by section.

### 9.2 Choose Region

Click the tile for your `<REGION>`. A datacenter sub-selector appears
(e.g. `FRA1`); if there is more than one, and your backend is already on a
specific datacenter, match it.

### 9.3 Choose an image

1. You will see tabs: **OS** · **Marketplace** · **Custom images** · **Backups &
   Snapshots**.
2. Click **Marketplace**.
3. In the search box, type `docker`.
4. Select **Docker on Ubuntu** (it will read something like *Docker 27.x.x on
   Ubuntu 24.04*). This gives you Docker Engine and the Compose plugin
   pre-installed, which saves a install step and a reboot.

> If for any reason Marketplace is unavailable, choose **OS → Ubuntu → 24.04
> (LTS) x64** and install Docker yourself afterwards with the official convenience
> script. The Marketplace image is simply less work.

### 9.4 Choose Size

1. Under **Droplet Type**, click **Basic** (this is "Shared CPU").
2. Under **CPU options**, click **Regular** (Disk type: SSD).
3. Scroll the size cards sideways until you find:

   > **$48/mo** · **8 GB RAM** / **4 vCPUs** · 160 GB SSD · 5 TB transfer

   Click that card.

**Why 4 vCPU / 8 GB is the floor, not a suggestion:** gunicorn runs 3 uvicorn
workers plus a separate worker container, `detector.startup_check()` loads ~387 MB
before the API accepts traffic, and the escalation path (AASIST + Wav2Vec2 +
RawNet3 fused by XGBoost) is CPU-bound. On 2 vCPU / 4 GB the API will OOM during
startup or time out its healthcheck.

**Do not pick a Premium or Dedicated tier** unless you have measured a need. The
extra cost buys clock speed you are not currently bottlenecked on.

### 9.5 Choose Authentication Method

1. Select **SSH Key**.
2. Tick the checkbox next to the key you added in Phase 8.

### 9.6 ⚠️ Advanced Options — the VPC

**This is the step people miss, and it cannot be fixed afterwards.**

1. Scroll down and click **Advanced Options** to expand it.
2. Find **VPC Network**.
3. From the dropdown, select the VPC from [Phase 3](#phase-3--create-the-vpc) —
   the one your backend droplet is on.

> Some layouts show VPC selection outside Advanced Options, in its own **Network**
> section. Either way, do not leave it on a default you have not checked.

### 9.7 Additional options

Still in this area, tick:

- ☑ **Add improved metrics monitoring and alerting (free)** — this installs the
  DO metrics agent, which you need for the alerts in
  [Phase 18](#phase-18--monitoring-and-alerts).

Leave these **unticked**:

- ☐ IPv6 — nothing here needs it, and it is one more surface to firewall
- ☐ User data — not used
- ☐ Backups — droplet backups are a paid add-on (~20% of droplet cost). The
  application-level backup in [Phase 16](#phase-16--nightly-encrypted-backups) is
  more useful because it is consistent and encrypted. Weekly manual snapshots are
  a cheaper second line of defence.

### 9.8 Finalize

1. **Quantity**: `1`
2. **Hostname**: `voiceguard-prod`
3. **Tags**: `voiceguard`, `production`
4. **Select Project**: `voiceguard-prod`
5. Click the big **Create Droplet** button at the bottom.

Provisioning takes 30–60 seconds. The progress bar completes and you land on the
droplet's page.

### 9.9 📝 Record the public IP

At the top of the droplet page is the **ipv4** address. Copy it. This is
`<PUBLIC_IP>` — you use it only for SSH, never for API traffic.

---

## Phase 10 — Find the private IP

This is the single most-mistyped value in the whole deployment. Get it from the
machine itself rather than from the panel, so there is no ambiguity.

### 10.1 💻 WORKSTATION — SSH in for the first time

```powershell
ssh root@<PUBLIC_IP>
```

- On first connect you get: *"The authenticity of host ... can't be established.
  Are you sure you want to continue connecting (yes/no)?"* — type `yes` and press
  Enter.
- Enter your SSH key passphrase if you set one.

You should land at a prompt like `root@voiceguard-prod:~#`.

### 10.2 🖥️ VG DROPLET — Read the private address

```bash
ip -4 addr show eth1 | awk '/inet /{print $2}'
```

Output looks like `10.114.0.3/20`. **The part before the slash is your
`VG_BIND_IP`** — `10.114.0.3` in this example.

If `eth1` does not exist, the droplet was created **without** a VPC. There is no
fix — destroy it and redo [Phase 9](#phase-9--create-the-droplet), being careful
at [9.6](#96--advanced-options--the-vpc).

Sanity checks:

```bash
# Should confirm Docker and the compose plugin are present
docker --version
docker compose version

# Should show 8 GB total
free -h

# Should show ~155 GB available on /
df -h /
```

### 10.3 🖥️ BACKEND — Get its private IP too

Open a **second terminal** and SSH into your backend droplet, then run the same
command:

```bash
ip -4 addr show eth1 | awk '/inet /{print $2}'
```

📝 **Write down:** `VG_BIND_IP` and `BACKEND_PRIVATE_IP`. Both must be inside the
VPC range you recorded in [3.3](#33--panel--record-the-ip-range). If they are not
in the same range, they are not on the same VPC, and nothing after this will work.

### 10.4 💻 WORKSTATION — Get your own public IP

You need this to restrict SSH access. In a browser, visit
**<https://ifconfig.me>** or run:

```powershell
(Invoke-WebRequest -Uri "https://ifconfig.me/ip").Content
```

📝 **Write down:** `ADMIN_IP`.

> If your home connection has a dynamic IP, this will change and lock you out of
> SSH. Either use a static IP, or accept a wider SSH rule (see
> [11.2](#112--panel--inbound-rules)).

---

## Phase 11 — Create the Cloud Firewall

This is the load-bearing security control. Everything else assumes it is correct.

### 11.1 🖱️ PANEL — Start the form

1. Left sidebar → **Networking**, then the **Firewalls** tab.
   Direct URL: **<https://cloud.digitalocean.com/networking/firewalls>**
2. Click **Create Firewall**.
3. **Name**: `voiceguard-prod-fw`

### 11.2 🖱️ PANEL — Inbound rules

DigitalOcean pre-fills an SSH rule. You will edit it and add one more.

**Rule 1 — SSH, restricted to you**

1. Find the pre-filled **SSH** row (`TCP`, port `22`).
2. In its **Sources** field, click the ✕ on **All IPv4** and **All IPv6** to remove
   them.
3. Type your `ADMIN_IP` followed by `/32` — e.g. `203.0.113.45/32` — and press
   Enter.

> If your IP is dynamic, use your ISP's range or a VPN exit IP instead. Leaving SSH
> open to `0.0.0.0/0` is survivable with key-only auth, but it is not what this
> design assumes.

**Rule 2 — the API port, restricted to your backend**

1. Click **New rule** → **Custom**.
2. **Type**: `Custom`
3. **Protocol**: `TCP`
4. **Port Range**: `8443`
5. **Sources**: type `<BACKEND_PRIVATE_IP>/32` — e.g. `10.114.0.2/32` — and press
   Enter.

> ⛔ **Do not** add `All IPv4` here. That would put the detector on the public
> internet, which the entire architecture exists to prevent.

**Delete every other inbound rule.** There should be exactly two rows when you are
done. In particular there is no HTTP (80) or HTTPS (443) rule — VoiceGuard serves
nothing publicly.

### 11.3 🖱️ PANEL — Outbound rules

Leave the defaults exactly as they are: **ICMP, All TCP, All UDP to All IPv4 and
All IPv6**.

The droplet needs outbound access to pull images from the registry, reach Spaces
for backups, and install OS updates. Locking outbound down further is possible but
is a separate exercise and will break `apt` and `docker pull` if done carelessly.

### 11.4 🖱️ PANEL — Apply to the droplet

1. Scroll to **Apply to Droplets**.
2. In the search field type `voiceguard-prod` and select it from the dropdown.
   (You can also apply by tag — type `voiceguard` — which auto-covers future
   droplets with that tag.)
3. Click **Create Firewall**.

### 11.5 💻 WORKSTATION — Verify it is actually closed

From your workstation — which is **not** the backend — confirm the API port is
unreachable:

```powershell
curl.exe -m 5 -k https://<PUBLIC_IP>:8443/ping
```

**This must time out or be refused.** If it returns anything at all, your firewall
is wrong. Stop and fix it before continuing.

Nothing is listening yet either, so a refusal now proves less than it will after
[Phase 13](#phase-13--configure-and-start-the-stack). **Run this check again at
go-live** — it is item 4 on the [Phase 19 checklist](#phase-19--go-live-checklist).

---

## Phase 12 — Build and push the image

Back on your own machine now.

### 12.1 💻 WORKSTATION — Set your variables

Open **Git Bash** in the repo root (right-click in the folder → *Git Bash Here*),
and export the Spaces build credentials from [Phase 5](#phase-5--create-spaces-access-keys):

```bash
export SPACES_KEY=DO00XXXXXXXXXXXXXXXX
export SPACES_SECRET=your-43-character-secret-here
export SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com
export SPACES_REGION=fra1
export SPACES_BUCKET=voiceguard-models-yourcompany
# SPACES_PREFIX defaults to voiceguard/model_store — leave it unless you changed it
```

These are the same five variables CI uses. They exist only on your workstation.

### 12.2 💻 WORKSTATION — Fill the build context

⚠️ **This step is mandatory.** Skip it and `docker build` copies an empty
`model_store`, and the container fails `startup_check()` on first boot.

```bash
python bundle_registry.py pull --active
python bundle_registry.py active
```

The second command must print `v9h`. Verify the bundle's integrity before you bake
it into an image:

```bash
python bundle_registry.py verify v9h
```

Confirm the files landed:

```bash
ls -la model_store/v9h/
```

You should see the model files and a manifest. If `pull --active` errors on a
missing variable, one of your five `SPACES_*` exports is wrong — the error names
which one.

### 12.3 💻 WORKSTATION — Log Docker into the registry

Make sure **Docker Desktop is running** first, then:

```bash
docker login registry.digitalocean.com \
  --username <DO_API_TOKEN> \
  --password <DO_API_TOKEN>
```

Yes — the token goes in **both** fields. That is how DOCR authentication works.
You should see `Login Succeeded`.

### 12.4 💻 WORKSTATION — Build

```bash
SHA=$(git rev-parse --short HEAD)
REG=registry.digitalocean.com/<REGISTRY_NAME>

docker build -t $REG/voiceguard:$SHA -t $REG/voiceguard:latest .
```

**The first build takes 15–40 minutes.** It installs torch and transformers, then
downloads and bakes in `facebook/wav2vec2-base`. Later builds are much faster —
the Dockerfile orders its layers so dependencies, the HF weights, and the model
bundle each cache separately from application code.

While it runs, note what the Dockerfile is doing:

- Sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so the running container
  **never reaches Hugging Face**
- Installs `ffmpeg`, which the detector shells out to for audio decode
- Copies `model_store/` as its own layer before the application code

### 12.5 💻 WORKSTATION — Push both tags

```bash
docker push $REG/voiceguard:$SHA
docker push $REG/voiceguard:latest
```

The first push uploads roughly 1.2–1.5 GB compressed. Later pushes only send
changed layers, which is usually just the application layer — tens of MB.

> ⛔ **Never overwrite a SHA tag.** Rollback is "deploy a previous SHA", and that
> only works if SHA tags are immutable. Always build a fresh commit.

### 12.6 If your upload is too slow

Pushing 1.5 GB over a home uplink can take hours. Two alternatives:

**Option A — build on the droplet.** SSH in, clone the repo there, export the
`SPACES_*` variables, and run the same commands. The droplet's uplink to the
registry is fast and free. The cost is that a build competes with production for
CPU and RAM — acceptable for the first deploy, not for routine ones.

**Option B — build on a temporary droplet.** Create a second `s-4vcpu-8gb` droplet
in the same region, build and push from it, then destroy it. Costs about $0.07/hour
and keeps production untouched. This is the cleanest option if you deploy often.

### 12.7 🖱️ PANEL — Confirm the image arrived

1. Left sidebar → **Container Registry**.
2. You should see a repository named **voiceguard** with two tags: your SHA and
   `latest`. Note the size — this is what counts against your plan quota.

---

## Phase 13 — Configure and start the stack

### 13.1 🖥️ VG DROPLET — Get the deploy files onto the box

`docker-compose.prod.yml` bind-mounts `./deploy/Caddyfile`, so the compose file and
the `deploy/` directory must sit together. Cloning the repo is simplest:

```bash
mkdir -p /srv/voiceguard && cd /srv/voiceguard
git clone https://github.com/SafeguardmediaHub/voiceguard.git .
```

If the repo is private, either use a deploy token in the URL, or copy just the
three things you need up from your workstation:

```powershell
# Run from the repo root on your WORKSTATION
scp docker-compose.prod.yml root@<PUBLIC_IP>:/srv/voiceguard/
scp -r deploy root@<PUBLIC_IP>:/srv/voiceguard/
scp .env.example root@<PUBLIC_IP>:/srv/voiceguard/
```

### 13.2 🖥️ VG DROPLET — Log Docker in

```bash
docker login registry.digitalocean.com \
  --username <DO_API_TOKEN> \
  --password <DO_API_TOKEN>
```

### 13.3 🖥️ VG DROPLET — Write the `.env`

```bash
cd /srv/voiceguard
cp .env.example .env
chmod 600 .env
nano .env
```

Set exactly these five values:

```ini
VOICEGUARD_IMAGE=registry.digitalocean.com/<REGISTRY_NAME>/voiceguard:<SHA>
VG_BIND_IP=10.114.0.3
VG_PORT=8443
WORKERS=3
VOICEGUARD_MAX_UPLOAD_MB=25
VOICEGUARD_ALLOWED_ORIGINS=
```

Line by line:

| Key | What to put | Why |
|---|---|---|
| `VOICEGUARD_IMAGE` | The **SHA tag**, not `latest` | Pinning to a SHA is what makes rollback deterministic |
| `VG_BIND_IP` | The private IP from [10.2](#102--vg-droplet--read-the-private-address) | Caddy binds here and nowhere else. `0.0.0.0` would expose the detector publicly |
| `VG_PORT` | `8443` | Must match the firewall rule from [11.2](#112--panel--inbound-rules) |
| `WORKERS` | `3` | 3 gunicorn workers on 4 vCPU leaves headroom for the worker container |
| `VOICEGUARD_MAX_UPLOAD_MB` | `25` | Must agree with `request_body max_size 25MiB` in `deploy/Caddyfile` |
| `VOICEGUARD_ALLOWED_ORIGINS` | *(empty)* | Empty means no browser origin is allowed — correct for server-to-server |

Save in nano with **Ctrl+O**, **Enter**, then **Ctrl+X**.

> ⛔ **Do not add `SPACES_*` to this file.** The bundle is baked into the image;
> the entrypoint only attempts a Spaces pull when `SPACES_BUCKET` is set, and
> production should hold no Spaces credentials at all.
>
> ⛔ **Do not add `VOICEGUARD_DEVICE`.** Production stays on the default CPU path.
> `tests/test_docker_context.py::test_deploy_config_never_sets_the_device_override`
> exists specifically to catch this.

### 13.4 🖥️ VG DROPLET — Pull and start

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

### 13.5 🖥️ VG DROPLET — Wait out the startup

**The `api` container will sit in `starting` for up to about 3 minutes.** This is
expected, not a fault. `api.py`'s lifespan runs `detector.startup_check()`, which
loads ~387 MB of weights and classifies a fixture clip before accepting any
traffic. That is why the healthcheck sets `start_period: 180s`.

Watch it happen:

```bash
docker compose -f docker-compose.prod.yml logs -f api
```

Press **Ctrl+C** to stop following. Then check status until `api` reads `healthy`:

```bash
docker compose -f docker-compose.prod.yml ps
```

Caddy waits on `service_healthy`, so it deliberately will not come up before the
API is genuinely ready.

> **The startup check fails closed.** A broken or tampered bundle stops the
> container rather than serving degraded verdicts. If `api` never turns healthy,
> go to [Troubleshooting](#troubleshooting).

### 13.6 🖥️ VG DROPLET — First real request

```bash
curl -sk https://localhost:8443/ping | python3 -m json.tool
```

You should get JSON reporting liveness and the active bundle version (`v9h`).

`-k` is needed here because Caddy serves a certificate from its own **internal
CA** for the hostname `voiceguard.internal`, and you are connecting to
`localhost`. From the backend, after [Phase 15](#phase-15--wire-up-the-backend),
`-k` will not be needed.

---

## Phase 14 — Issue the backend an API key

### 14.1 🖥️ VG DROPLET — Create it

```bash
cd /srv/voiceguard
docker compose -f docker-compose.prod.yml exec api \
  python auth.py create --client "backend-prod"
```

### 14.2 ⚠️ Copy the plaintext key immediately

Keys are SHA-256 hashed into `auth_keys.json` on the `vg-data` volume. **The
plaintext is printed once and is not recoverable.** Put it straight into your
backend's secret manager.

List and revoke:

```bash
docker compose -f docker-compose.prod.yml exec api python auth.py list
docker compose -f docker-compose.prod.yml exec api python auth.py revoke <key_id>
```

`auth.py create` also takes repeatable `--scope` flags if you want to constrain a
key. Issue one key per calling system — that way revoking one does not take down
the others.

---

## Phase 15 — Wire up the backend

Two things have to be true on the backend: it must **resolve** `voiceguard.internal`,
and it must **trust** Caddy's internal CA. Skipping either is the most common
integration failure.

### 15.1 🖥️ VG DROPLET — Export the root certificate

```bash
cd /srv/voiceguard
docker compose -f docker-compose.prod.yml exec caddy \
  cat /data/caddy/pki/authorities/local/root.crt > voiceguard-root.crt
cat voiceguard-root.crt
```

Copy the entire block including the `-----BEGIN CERTIFICATE-----` and
`-----END CERTIFICATE-----` lines.

### 15.2 🖥️ BACKEND — Add the hostname

```bash
echo "<VG_BIND_IP>  voiceguard.internal" >> /etc/hosts
ping -c 1 voiceguard.internal
```

Use the **private** IP. The certificate is issued for `voiceguard.internal`, so the
backend must reach it by that name or TLS verification fails on hostname mismatch.

### 15.3 🖥️ BACKEND — Install and trust the CA

```bash
nano /usr/local/share/ca-certificates/voiceguard-root.crt
```

Paste the certificate, save (**Ctrl+O**, **Enter**, **Ctrl+X**), then:

```bash
update-ca-certificates
```

You should see `1 added`.

### 15.4 🖥️ BACKEND — Verify end to end

```bash
curl https://voiceguard.internal:8443/ping
```

**No `-k`.** If this succeeds without it, DNS and trust are both correct. If you
get a certificate error, redo 15.2–15.3.

> ⛔ The CA lives in the `voiceguard_caddy-data` volume. **Never delete that
> volume.** A new CA would be generated and every backend that trusted the old
> root would start failing verification until you re-export and re-trust.

### 15.5 The request contract

Detection is **asynchronous** — submit, then poll.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/ping` | none | Liveness + active bundle version |
| `POST` | `/detect` | Bearer | Submit audio → `202 {job_id, status_url}` |
| `GET` | `/jobs/{job_id}` | Bearer | Poll status/result, scoped to the calling key |
| `GET` | `/drift`, `/drift/latest`, `/drift/history`, `/drift/baseline` | Bearer | Drift monitor reads |
| `GET` | `/` | none | Demo HTML page |

Test it:

```bash
KEY=<the plaintext key from Phase 14>

curl -X POST https://voiceguard.internal:8443/detect \
     -H "Authorization: Bearer $KEY" \
     -F "file=@sample.wav"
# → 202 {"job_id":"...","status_url":"/jobs/..."}

curl -H "Authorization: Bearer $KEY" \
     https://voiceguard.internal:8443/jobs/<job_id>
```

`scripts/voiceguard_client.py` in the repo is a reference client that already
handles polling, `429` + `Retry-After`, and pre-flight size rejection. Start from
it rather than writing your own.

**The 25 MB cap is enforced in three places** — your backend client, Caddy's
`request_body max_size 25MiB`, and `VOICEGUARD_MAX_UPLOAD_MB`. All three must
agree, and note that Caddy uses **MiB** (26,214,400 bytes) deliberately: `25MB`
would be parsed as decimal and reject uploads the API would have accepted.

---

## Phase 16 — Nightly encrypted backups

`deploy/backup.py` snapshots `jobs.db` through SQLite's **online backup API** — a
plain file copy of a WAL-mode database is not consistent — plus `auth_keys.json`
and `governance/audit_log.jsonl`, optionally Fernet-encrypted, to Spaces.

### 16.1 💻 WORKSTATION — Generate the encryption key

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

This prints a 44-character key.

⚠️ **Store this outside the droplet** — password manager, or a sealed envelope in a
safe. Without it your backups are unreadable, and a backup you cannot restore is
not a backup.

### 16.2 🖥️ VG DROPLET — Write the credentials file

These credentials deliberately live **outside** `.env`, because `.env` is loaded
into the API and worker containers and they have no business holding Spaces keys.
Use the **backup** key pair from [5.3](#53--panel--generate-a-second-key-for-backups).

```bash
mkdir -p /etc/voiceguard
nano /etc/voiceguard/backup.env
```

Paste:

```ini
SPACES_KEY=DO00YYYYYYYYYYYYYYYY
SPACES_SECRET=your-backup-key-secret
SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com
SPACES_REGION=fra1
SPACES_BUCKET=voiceguard-backups-yourcompany
VOICEGUARD_BACKUP_KEY=your-44-char-fernet-key
```

Then lock it down:

```bash
chmod 600 /etc/voiceguard/backup.env
```

### 16.3 🖥️ VG DROPLET — Run it once by hand

Never schedule a job you have not watched succeed.

```bash
set -a; . /etc/voiceguard/backup.env; set +a
python3 /srv/voiceguard/deploy/backup.py \
  --data-dir /var/lib/docker/volumes/voiceguard_vg-data/_data
```

It should print the keys it uploaded. If `boto3` or `cryptography` is missing:

```bash
apt-get update && apt-get install -y python3-pip
pip3 install boto3 cryptography --break-system-packages
```

> The volume path is `voiceguard_vg-data` because `docker-compose.prod.yml` pins
> `name: voiceguard`. That is deliberate — deriving the name from the directory
> would silently break this cron job if anyone moved the checkout.

### 16.4 🖥️ VG DROPLET — Schedule it

```bash
cat >/etc/cron.d/voiceguard-backup <<'EOF'
0 2 * * * root set -a; . /etc/voiceguard/backup.env; set +a; /usr/bin/python3 /srv/voiceguard/deploy/backup.py --data-dir /var/lib/docker/volumes/voiceguard_vg-data/_data >>/var/log/voiceguard-backup.log 2>&1
EOF
chmod 644 /etc/cron.d/voiceguard-backup
```

Check it ran the next morning:

```bash
tail -50 /var/log/voiceguard-backup.log
```

### 16.5 🖱️ PANEL — Weekly snapshots as a second line

1. Left sidebar → **Droplets** → click **voiceguard-prod**.
2. Click the **Snapshots** tab.
3. Click **Take Snapshot**, name it `voiceguard-YYYY-MM-DD`.

Snapshots capture the whole disk including Docker volumes, so they recover from
"the droplet is gone" in a way the file-level backup does not. Do one now, before
go-live, and repeat weekly. Snapshots are billed per GB stored.

### 16.6 What each volume holds

| Volume | Holds | Losing it means |
|---|---|---|
| `voiceguard_vg-data` | `jobs.db`, `auth_keys.json`, `governance/audit_log.jsonl`, `drift/`, transient uploads | Every client API key is revoked; the audit chain of custody is broken |
| `voiceguard_caddy-data` | Caddy's internal CA + issued certs | Every backend that trusted the old root fails TLS until re-pinned |
| `voiceguard_caddy-config` | Caddy autosave config | Nothing important |

---

## Phase 17 — Drift monitoring

> ⚠️ **Read this before promising drift monitoring to anyone.** The **reader** is
> wired and works out of the box. The **producer** needs a validation set that is
> not in the image and that you must supply. Until you do, `/drift` returns empty
> sections and **no drift alerting happens at all**.

### 17.1 Why the shipped validation set does not work

`models/val_v8_fresh.json` contains **absolute Windows paths from a development
laptop**. Those cannot resolve inside a Linux container. You need a regenerated
manifest with container paths.

### 17.2 🖥️ VG DROPLET — Stage the validation set

Copy your labelled clips to `/srv/voiceguard/valset/` and write a `val.json` whose
`path` fields are the **container** paths:

```json
[
  {"path": "/valset/real/clip_0001.mp3", "label": 0, "source": "real_local"},
  {"path": "/valset/fake/noizai_0001.mp3", "label": 1, "source": "noizai_tts"}
]
```

- `label`: `0` = real, `1` = fake
- `source`: groups clips for the per-source catch-rate alerts

All three keys are **required** — `_validate_manifest_schema` rejects the manifest
otherwise.

### 17.3 🖥️ VG DROPLET — Establish the baseline

```bash
cd /srv/voiceguard
docker compose -f docker-compose.prod.yml run --rm \
  -v /srv/voiceguard/valset:/valset:ro \
  -e DRIFT_VAL_MANIFEST=/valset/val.json \
  api python drift_monitor_3.py --init-baseline
```

### 17.4 🖥️ VG DROPLET — Schedule the nightly run

03:30 UTC — off the CI schedule and off peak.

```bash
cat >/etc/cron.d/voiceguard-drift <<'EOF'
30 3 * * * root cd /srv/voiceguard && /usr/bin/docker compose -f docker-compose.prod.yml run --rm -v /srv/voiceguard/valset:/valset:ro -e DRIFT_VAL_MANIFEST=/valset/val.json api python drift_monitor_3.py --run >>/var/log/voiceguard-drift.log 2>&1
EOF
chmod 644 /etc/cron.d/voiceguard-drift
```

A run writes `drift_report_<ts>.json`, appends `drift_log.jsonl`, and updates
`drift_alert_state.json` — all in `/data/drift` on the `vg-data` volume, all
immediately visible on the `/drift` endpoint and surviving a redeploy.

### 17.5 How alerting behaves

| Signal | Threshold | Notes |
|---|---|---|
| Clean ensemble EER | ±3.0 pp | Also needs `p < 0.05` on a two-proportion z-test |
| Deployed (cascade) EER | ±3.0 pp | What production actually ships — can regress while the ensemble looks fine |
| Per-source catch rate | −10 pp | −15 pp for phone-class sources |
| Noiz.ai catch rate | −15 pp | |
| Val manifest hash | Any change | You changed the test set — re-baseline deliberately |

A breach does **not** fire immediately. `DRIFT_CONFIRM_RUNS` (default `2`) requires
consecutive breaching runs, and a clean run resets the counter. Only confirmed
alerts write `retrain_trigger.json`, which surfaces on `/drift` as
`retrain.retrain_needed`.

### 17.6 Optional email alerts

Add to `/srv/voiceguard/.env` and restart:

```ini
DRIFT_SMTP_HOST=smtp.example.com
DRIFT_SMTP_PORT=587
DRIFT_SMTP_USER=alerts@example.com
DRIFT_SMTP_PASS=...
DRIFT_ALERT_TO=ops@example.com,michael@example.com
```

> **Note:** DigitalOcean blocks outbound SMTP on port 25 by default for new
> accounts. Use port 587 with a relay (SendGrid, Mailgun, Postmark), not a direct
> MX connection.

After acting on a trigger:

```bash
docker compose -f docker-compose.prod.yml run --rm api python drift_monitor_3.py --retrain-status
docker compose -f docker-compose.prod.yml run --rm api python drift_monitor_3.py --clear-trigger
```

---

## Phase 18 — Monitoring and alerts

### 18.1 🖱️ PANEL — Create alert policies

1. Left sidebar under **MONITORING**, click **Alerts**.
   Direct URL: **<https://cloud.digitalocean.com/monitoring/alerts>**
2. Click **Create Alert Policy**.

Create three, repeating the flow each time:

**Alert 1 — CPU**
- **Metric**: `Droplet` → `CPU`
- **Condition**: `is above` `80` `%` for `10 minutes`
- **Apply to**: select `voiceguard-prod` (or the `voiceguard` tag)
- **Alert name**: `VoiceGuard CPU high`
- **Send notifications to**: your email, and Slack if you have it connected
- Click **Create Alert Policy**

**Alert 2 — Memory**
- Same flow, **Metric**: `Memory`, **is above** `85%` for `10 minutes`
- Name: `VoiceGuard memory high`

**Alert 3 — Disk**
- Same flow, **Metric**: `Disk Utilization`, **is above** `80%` for `10 minutes`
- Name: `VoiceGuard disk high`

These require the metrics agent, which you enabled at
[9.7](#97--additional-options). If the alert page says no droplets have monitoring,
SSH in and run:

```bash
curl -sSL https://repos.insights.digitalocean.com/install.sh | bash
```

### 18.2 Uptime check

Your **backend** should poll `/ping` on a schedule and alert if it fails. DO's
built-in **Uptime** checks reach from the public internet, which cannot see
VoiceGuard by design — so this one has to live in your own monitoring.

### 18.3 Log rotation is already handled

The compose file caps Docker logs at 10 MB × 3 files per service, so they cannot
fill the disk. Nothing to configure.

---

## Phase 19 — Go-live checklist

Work through every line. Do not skip the negative tests — they are the ones that
catch a misconfigured firewall.

**Infrastructure**

- [ ] Droplet is `4 vCPU / 8 GB` and shows `healthy` for both `api` and `worker`
- [ ] Droplet, backend, and VPC are all in the same region
- [ ] `ip -4 addr show eth1` returns an address inside the VPC range
- [ ] Cloud Firewall has **exactly two** inbound rules: `22` from `<ADMIN_IP>/32`, `8443` from `<BACKEND_PRIVATE_IP>/32`

**Negative tests — these must FAIL**

- [ ] From your workstation: `curl -m 5 -k https://<PUBLIC_IP>:8443/ping` **times out**
- [ ] From any third machine: SSH to the droplet is **refused**
- [ ] `curl` to `/detect` **without** an `Authorization` header returns **401**

**Positive tests — these must SUCCEED**

- [ ] On the droplet: `curl -sk https://localhost:8443/ping` returns JSON with bundle `v9h`
- [ ] On the backend: `curl https://voiceguard.internal:8443/ping` succeeds **without `-k`**
- [ ] A real `POST /detect` returns `202` with a `job_id`
- [ ] Polling `/jobs/{job_id}` eventually returns a verdict
- [ ] An upload over 25 MB is rejected with `413`

**Operations**

- [ ] `.env` is `chmod 600` and pinned to a **SHA tag**, not `latest`
- [ ] `.env` contains **no** `SPACES_*` and **no** `VOICEGUARD_DEVICE`
- [ ] `/etc/voiceguard/backup.env` is `chmod 600`
- [ ] Backup ran manually and printed uploaded keys
- [ ] `VOICEGUARD_BACKUP_KEY` is stored **off the droplet**
- [ ] Both cron files exist in `/etc/cron.d/`
- [ ] A pre-go-live snapshot exists
- [ ] Three monitoring alert policies are active
- [ ] The API key plaintext is in the backend's secret manager and **nowhere else**
- [ ] Caddy's root cert is trusted on the backend and archived somewhere safe

**Known gaps to state explicitly to stakeholders**

- [ ] Drift monitoring is **inactive** until you supply a validation set ([Phase 17](#phase-17--drift-monitoring))
- [ ] There is **no automated deploy**. `.github/workflows/ci.yml` runs tests only — build, push and deploy are manual
- [ ] **Single point of failure.** One droplet, vertical scaling only. The next step if load grows is moving the SQLite queue to managed Postgres and adding a second droplet
- [ ] **Rate limiting is per-process.** `request_protection.py` keeps counters in memory, so the effective limit scales with worker count and does not aggregate across them

---

## Day-2 operations

### Deploy a new version

```bash
# WORKSTATION — build and push the new SHA
SHA=$(git rev-parse --short HEAD)
REG=registry.digitalocean.com/<REGISTRY_NAME>
docker build -t $REG/voiceguard:$SHA .
docker push $REG/voiceguard:$SHA
```

```bash
# VG DROPLET — point at it and restart
cd /srv/voiceguard
sed -i "s|^VOICEGUARD_IMAGE=.*|VOICEGUARD_IMAGE=registry.digitalocean.com/<REGISTRY_NAME>/voiceguard:$NEW_SHA|" .env
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps      # wait for api = healthy
curl -sk https://localhost:8443/ping
```

Expect ~3 minutes of downtime while the new API completes its startup check.

### Roll back

Point `VOICEGUARD_IMAGE` at the previous SHA tag and repeat. This is the whole
reason SHA tags are never overwritten.

### Scale the worker if the queue backs up

```bash
docker compose -f docker-compose.prod.yml up -d --scale worker=2
```

This is safe: `jobs.claim_next` uses `BEGIN IMMEDIATE`, and `governance.AuditLog.append`
takes an OS file lock, so concurrent writers cannot break the hash chain.

### Update the model bundle

Promotion is a production change and is **attributable by design**:

```bash
python bundle_registry.py promote v9h2 --actor "firstname.lastname" --reason "H1 eval passed"
```

- `--actor` must be a real name. `cli`, `admin`, `root`, `unknown`, and anything
  under 3 characters are rejected by `_require_named_actor` — ISO 42001 change
  control, and it lands in the tamper-evident hash chain in `ACTIVE.json`.
- A version in `BLOCKED_VERSIONS` can never be activated, and `rollback` refuses to
  land on one. **`v9` is blocked**: its `aasist.pt` is the collapsed from-scratch
  V9 architecture that the current `detector.py` cannot load, so activating it
  would crash the service on start.
- Promotion runs the sub-model health gate against `tests/probe_clips/`, which is
  baked into the image precisely so the gate can certify in production. Avoid
  `--skip-health` — it exists for emergencies, not routine use.

Then rebuild ([Phase 12](#phase-12--build-and-push-the-image)) and redeploy so the
new bundle is baked in.

```bash
python bundle_registry.py rollback --actor "firstname.lastname" --reason "regression in prod"
python bundle_registry.py active
python bundle_registry.py verify v9h
python bundle_registry.py log v9h
```

### Restore from backup

```bash
docker compose -f docker-compose.prod.yml down

# Decrypt with VOICEGUARD_BACKUP_KEY if it was set, then:
cp jobs.db auth_keys.json /var/lib/docker/volumes/voiceguard_vg-data/_data/
mkdir -p /var/lib/docker/volumes/voiceguard_vg-data/_data/governance
cp audit_log.jsonl /var/lib/docker/volumes/voiceguard_vg-data/_data/governance/

docker compose -f docker-compose.prod.yml up -d
```

⚠️ Restore `audit_log.jsonl` **byte for byte**. The tamper-evident hash chain is
computed over exact bytes, and any line-ending translation — which Windows tooling
does silently — breaks verification permanently.

### Rotate an API key

```bash
docker compose -f docker-compose.prod.yml exec api python auth.py create --client "backend-prod-v2"
# deploy the new key to the backend, confirm traffic is flowing, THEN:
docker compose -f docker-compose.prod.yml exec api python auth.py list
docker compose -f docker-compose.prod.yml exec api python auth.py revoke <old_key_id>
```

### Routine health commands

```bash
docker compose -f docker-compose.prod.yml logs -f api worker caddy
docker compose -f docker-compose.prod.yml exec api python auth.py list
docker compose -f docker-compose.prod.yml run --rm api python bundle_registry.py active
curl -sk https://localhost:8443/ping | python3 -m json.tool
df -h /
docker system df
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `api` never becomes healthy; logs show a bundle error | `startup_check()` failing closed on a missing or tampered bundle | `docker compose -f docker-compose.prod.yml run --rm api python bundle_registry.py verify v9h`. Usually you skipped [12.2](#122--workstation--fill-the-build-context) — rebuild with a fresh `pull --active` |
| `api` healthy but Caddy will not start | Caddy waits on `service_healthy` | Wait out the 180 s `start_period`, then `logs api` |
| Backend gets a TLS verification error | Caddy's internal root not trusted, or `caddy-data` was recreated | Redo [Phase 15](#phase-15--wire-up-the-backend) |
| Backend gets connection refused | Working as designed | Traffic must originate from the backend's **private** IP inside the VPC |
| `docker compose up` says `VG_BIND_IP` is unset | `.env` missing or in the wrong directory | It must sit beside `docker-compose.prod.yml` in `/srv/voiceguard` |
| `Cannot assign requested address` on start | `VG_BIND_IP` is not an address this droplet actually holds | Re-run [10.2](#102--vg-droplet--read-the-private-address) and correct `.env` |
| `413` on an upload the API would accept | MiB vs MB mismatch | `deploy/Caddyfile` must say `25MiB`, matching `VOICEGUARD_MAX_UPLOAD_MB=25` |
| `/drift` always empty | No monitor run yet, or `DRIFT_OUTPUT_DIR` overridden off `/data` | [Phase 17](#phase-17--drift-monitoring). `curl .../drift` reports the `output_dir` it actually read |
| Jobs queue up, results slow | One worker saturated | `docker compose -f docker-compose.prod.yml up -d --scale worker=2` |
| Disk filling | Old images, or drift reports accumulating | `docker image prune -a --filter "until=720h"`; prune old `drift_report_*` in `/data/drift`. The worker already deletes inputs after processing |
| `docker pull` says unauthorized | Registry login expired | Re-run `docker login` ([13.2](#132--vg-droplet--log-docker-in)); if the token expired, generate a new one ([Phase 7](#phase-7--create-an-api-token)) |
| Push rejected, quota exceeded | Registry plan too small | [6.1](#61--read-this-before-you-pick-a-plan). Upgrade the plan, or delete old tags in the panel under **Container Registry** |
| Locked out of SSH | Your `ADMIN_IP` changed | Panel → **Networking → Firewalls → voiceguard-prod-fw → Rules**, update the SSH source. The panel's **Droplet Console** (Access tab) also gets you in regardless of firewall |
| Build OOMs on the droplet | 8 GB is tight for a torch build alongside production | Build on a temporary droplet instead — [12.6](#126-if-your-upload-is-too-slow) Option B |

### Reading the logs

```bash
docker compose -f docker-compose.prod.yml logs --tail=200 api
docker compose -f docker-compose.prod.yml logs --tail=200 worker
docker compose -f docker-compose.prod.yml logs --tail=200 caddy
docker inspect --format='{{json .State.Health}}' $(docker compose -f docker-compose.prod.yml ps -q api) | python3 -m json.tool
```

### Emergency console access

If SSH is unreachable for any reason: Panel → **Droplets** → **voiceguard-prod** →
**Access** tab → **Launch Droplet Console**. This is an out-of-band browser
terminal that bypasses the firewall entirely. Set a root password there first
(**Reset Root Password**) if you have never used it.

---

## Appendix A — What it costs

Approximate monthly list prices. Verify current figures on DigitalOcean's pricing
page — these move.

| Item | Spec | Cost |
|---|---|---|
| Droplet | `s-4vcpu-8gb`, Basic/Regular | **$48** |
| Container Registry | Professional, 100 GiB | **$20** |
| Spaces (models) | 250 GB + 1 TB transfer | **$5** |
| Spaces (backups) | 250 GB + 1 TB transfer | **$5** |
| VPC | — | Free |
| Cloud Firewall | — | Free |
| Monitoring + alerts | — | Free |
| Snapshots | ~10 GB stored | ~$1 |
| **Total** | | **~$79/month** |

Ways to trim:

- **Registry Basic instead of Professional**: saves $15/mo, costs you tag history.
  Only sensible if you prune aggressively and accept a narrow rollback window.
- **One Spaces bucket instead of two**: saves $5/mo, at the cost of letting the
  production backup credentials also reach your model bundles. Not a trade I would
  make.
- **Droplet backups**: leave them off. Application-level backups plus weekly
  snapshots cover the same ground for less.

---

## Appendix B — Every value you need to write down

Fill this in as you go. Keep it in a password manager, not a text file.

| # | Value | Where it comes from | Yours |
|---|---|---|---|
| 1 | Region slug | [Phase 2](#phase-2--pick-your-region-and-stick-to-it) | |
| 2 | VPC name | [Phase 3](#phase-3--create-the-vpc) | |
| 3 | VPC IP range | [3.3](#33--panel--record-the-ip-range) | |
| 4 | `SPACES_BUCKET` (models) | [4.1](#41--panel--create-the-bucket) | |
| 5 | `SPACES_BUCKET` (backups) | [4.3](#43--panel--consider-a-second-bucket-for-backups) | |
| 6 | `SPACES_ENDPOINT` | [Phase 4](#phase-4--create-the-spaces-bucket) | |
| 7 | `SPACES_KEY` / `SPACES_SECRET` — build | [5.1](#51--panel--generate-the-key-pair) | |
| 8 | `SPACES_KEY` / `SPACES_SECRET` — backup | [5.3](#53--panel--generate-a-second-key-for-backups) | |
| 9 | `REGISTRY_NAME` | [6.2](#62--panel--create-it) | |
| 10 | `DO_API_TOKEN` | [Phase 7](#phase-7--create-an-api-token) | |
| 11 | `<PUBLIC_IP>` | [9.9](#99--record-the-public-ip) | |
| 12 | `VG_BIND_IP` (private) | [10.2](#102--vg-droplet--read-the-private-address) | |
| 13 | `BACKEND_PRIVATE_IP` | [10.3](#103--backend--get-its-private-ip-too) | |
| 14 | `ADMIN_IP` | [10.4](#104--workstation--get-your-own-public-ip) | |
| 15 | Deployed image SHA tag | [12.4](#124--workstation--build) | |
| 16 | Backend API key (plaintext) | [14.1](#141--vg-droplet--create-it) | |
| 17 | `VOICEGUARD_BACKUP_KEY` (Fernet) | [16.1](#161--workstation--generate-the-encryption-key) | |
| 18 | Caddy root cert (PEM) | [15.1](#151--vg-droplet--export-the-root-certificate) | |

**Items 7, 8, 10, 16 and 17 are secrets.** Items 16 and 17 are shown exactly once
and cannot be recovered — losing 17 makes every backup unreadable.

---

## Related documents

| Document | What it is |
|---|---|
| [`DEPLOYMENT.md`](../DEPLOYMENT.md) | The terse `doctl` operator runbook. Authoritative on application details |
| [`docs/API_REFERENCE.md`](API_REFERENCE.md) | Endpoint contracts for backend integrators |
| [`docs/CI-and-model-store.md`](CI-and-model-store.md) | How the bundle registry and Spaces layout work |
| [`docs/RUNBOOK-model-flow.md`](RUNBOOK-model-flow.md) | Model promotion and the health gate |
| [`docs/GRC_CONTROL_PACK.md`](GRC_CONTROL_PACK.md) | Risk register and controls — read §4 before an enterprise sale |
| [`docs/MODEL_CARD.md`](MODEL_CARD.md) | What the model does and its measured limits |
