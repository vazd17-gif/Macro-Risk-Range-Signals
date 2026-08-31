# Going live — VS Code and GitHub

Everything below is done once. After that the machine does the work.

---

## 1. Open the project in VS Code

Launch VS Code → **File ▸ Open Folder** → `C:\Users\vazd1\OneDrive\Desktop\Fractal Build`.

The repo is already initialised with a first commit, so the Source Control panel
(third icon in the left rail, or `Ctrl+Shift+G`) will show a clean tree.

Two extensions worth installing (`Ctrl+Shift+X`, search by name):

| Extension | Why |
|---|---|
| **Python** (Microsoft) | Run and debug any file with F5; the terminal picks up the right interpreter |
| **GitLens** | Shows when each line last changed — useful once parameters start being re-fitted |

The integrated terminal (`` Ctrl+` ``) is where every command below runs.

---

## 2. Put it on GitHub

**In GitHub Desktop:** *File ▸ Add local repository* → choose the project folder →
**Publish repository**.

**Make it private** unless you want the fitted parameters and your position book
public. `fractal/data/portfolio.csv` is tracked, so your holdings go with it.

What is deliberately *not* committed (see `.gitignore`):

- `fractal/cache/` — price history, re-downloaded on demand, changes daily
- `fractal/out/logs/`, `*.csv`, `*.eml`, `.last_sent` — run artefacts
- Nothing containing the app password. It lives only in your Windows user
  environment, which is why it survives a `git push` without ever being in a file.

---

## 3. Turn on GitHub Pages

The dashboard is copied to `docs/index.html` on every live run. Point Pages at it:

1. On github.com, open the repo → **Settings ▸ Pages**
2. **Source:** Deploy from a branch
3. **Branch:** `main`, **Folder:** `/docs` → Save

A minute later the URL appears at the top of that page:
`https://<your-username>.github.io/<repo-name>/`

That is the address to bookmark on your phone. It updates itself whenever the
intraday job pushes.

**If the repo is private,** Pages needs GitHub Pro. If you would rather not pay,
either make the repo public (and first move `portfolio.csv` out of it), or keep
using the Artifact link, which is private and needs no plan.

---

## 4. Schedule the two jobs

From the VS Code terminal, in the project folder:

```powershell
# Newsletter: weekdays 12:00 London. Refreshes the model first, then emails.
powershell -ExecutionPolicy Bypass -File automation\install_task.ps1

# Dashboard: every 10 minutes, 14:30-21:00 London (the US cash session).
powershell -ExecutionPolicy Bypass -File automation\install_live_task.ps1
```

Check either one:

```powershell
Get-ScheduledTaskInfo -TaskName FractalRiskRangeDaily
Start-ScheduledTask   -TaskName FractalRiskRangeDaily   # run it now
```

Logs land in `fractal/out/logs/`.

---

## How the two jobs differ

This is the part worth understanding, because it explains why the dashboard can be
live without the levels ever moving mid-session.

**Daily (12:00 London)** — `etf_report --sync --push` refits everything from the last completed
session: RANGE, TRADE, TREND, and the volume baselines. At noon London the most
recent close is yesterday's 16:00 ET, so the newsletter recaps it and carries the
levels for the session opening at 14:30. Then it emails.

**Intraday (every 10 min)** — re-prices only. The levels for a session are set by
the previous close and *do not change while it trades*; that is what makes them
tradeable. So the job fetches quotes, recomputes where spot sits relative to the
fixed levels, and rewrites the page. One batch request rather than 191 model fits.

That difference is why the dashboard can carry a **Crossed a line today** strip.
The daily run only ever sees closes, so a break that happens at 10am is invisible
to it until tonight. The live pass compares spot against the close-based state and
surfaces exactly those crossings at the top of the page.

The page also sets a 5-minute meta refresh, so a phone left open keeps current
without being touched.

---

## Day-to-day

```bash
python -m fractal.app.etf_report --sync      # full refresh (levels + newsletter files)
python -m fractal.app.etf_report --live      # re-price now
python -m fractal.app.publish --to you@x.com # draft a .eml to review
python -m fractal.app.portfolio status       # book vs today's signals
```

To commit changes: Source Control panel → type a message → **Commit** → **Sync**.

Two habits worth keeping. Re-run `python -m fractal.calib.validate_hedgeye`
whenever new Hedgeye levels arrive by email — it is the only check that the
reconstruction still tracks. And commit after re-fitting parameters, so a fit that
turns out worse can be reverted rather than argued about.

---

## Phone app and alerts

The dashboard installs to a phone home screen, and alerts are pushed by the same
scheduled job that re-prices it.

### Install the app

Open the site on the phone and use the browser menu:

- **iPhone** (Safari) — Share → **Add to Home Screen**
- **Android** (Chrome) — ⋮ → **Install app**

It gets its own icon and opens without browser chrome. The page reloads itself
every five minutes, so an open app stays current.

Push notifications do **not** come from the installed page. That would need a
server holding a subscription for each device, and GitHub Pages only serves static
files. Alerts go over ntfy instead — see below.

### Turn on alerts

```bash
python -m fractal.app.alerts --setup
```

That prints a private topic and the two `setx` commands to run. Install the
**ntfy** app (App Store / Play Store), subscribe to that topic, then:

```bash
python -m fractal.app.alerts --test
```

Pushover works too, if preferred: set `FRACTAL_PUSHOVER_TOKEN` and
`FRACTAL_PUSHOVER_USER` and it is used alongside or instead of ntfy.

### What fires, and what does not

| Event | Pushed? |
|---|---|
| TRADE or TREND crossed during the session | yes, high priority |
| A new ADD LONG / ADD SHORT / REMOVE LONG / COVER SHORT | yes |
| The daily signal set at noon | no — that is the newsletter |
| The same event, on the next ten-minute pass | no |
| Volatility indices, cash-like names | never |

The noon job **seeds** the day's signals as already-known instead of sending them;
pushing forty at once would just be a second copy of the email. Sent keys live in
`fractal/out/.alerted` and clear when the session date changes, so each distinct
event buzzes once. More than four new at once arrive as one digest.

> The ntfy topic name is the only thing protecting the channel — anyone who knows
> it can read your alerts. Keep it out of screenshots and shared repos.
