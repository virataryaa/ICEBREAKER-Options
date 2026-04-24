"""
run.py -- ICEBREAKER Options daily automator
Runs KC + CC ingest, commits updated parquets + atm.json, pushes to GitHub, sends email.
"""

import subprocess
import sys
import datetime
import traceback
from pathlib import Path

import win32com.client

# -- Paths --------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parent.parent
INGEST_KC    = ROOT / "Code" / "Ingest.py"
INGEST_CC    = ROOT / "Code" / "CC_Ingest.py"
INGEST_SB    = ROOT / "Code" / "SB_Ingest.py"
PARQUET_KC   = ROOT / "Database" / "KC_options_ice.parquet"
PARQUET_CC   = ROOT / "Database" / "CC_options_ice.parquet"
PARQUET_SB   = ROOT / "Database" / "SB_options_ice.parquet"
ATM_JSON     = ROOT / "Dashboard" / "atm.json"
LOG_FILE     = Path(__file__).resolve().parent / "run_log.txt"
PYTHON       = sys.executable

EMAIL_TO = "virat.arya@etgworld.com"

# -- Helpers ------------------------------------------------------------------

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def send_email(subject: str, body: str):
    try:
        ol   = win32com.client.Dispatch("Outlook.Application")
        mail = ol.CreateItem(0)
        mail.To      = EMAIL_TO
        mail.Subject = subject
        mail.Body    = body
        mail.Send()
        log("Email sent.")
    except Exception as e:
        log(f"Email failed: {e}")


def run_ingest(script: Path, label: str) -> tuple[bool, str]:
    log(f"Running {label} ingest...")
    result = subprocess.run(
        [PYTHON, str(script)],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def git_push(files: list[Path]) -> tuple[bool, str]:
    rel = [str(f.relative_to(ROOT)) for f in files if f.exists()]
    if not rel:
        return False, "No files to stage"
    cmds = [
        ["git", "add"] + rel,
        ["git", "commit", "-m",
         f"auto: daily options update {datetime.date.today()}"],
        ["git", "push"],
    ]
    out = ""
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        out += r.stdout + r.stderr
        if r.returncode != 0 and "nothing to commit" not in r.stderr:
            return False, out
    return True, out


# -- Main ---------------------------------------------------------------------

def main():
    today = datetime.date.today().isoformat()
    log("=" * 50)
    log(f"Options ingest started — {today}")

    ingests = [
        (INGEST_KC, "KC"),
        (INGEST_CC, "CC"),
        (INGEST_SB, "SB"),
    ]

    all_output = {}
    any_failed = False

    for script, label in ingests:
        ok, out = run_ingest(script, label)
        all_output[label] = (ok, out)
        log(f"{label} ingest: {'OK' if ok else 'FAILED'}")
        for line in out.strip().splitlines():
            log(f"  {line}")
        if not ok:
            any_failed = True

    if any_failed:
        failed_labels = [lbl for lbl, (ok, _) in all_output.items() if not ok]
        combined = "\n\n".join(
            f"=== {lbl} ===\n{out}" for lbl, (ok, out) in all_output.items()
        )
        send_email(
            f"[ICEBREAKER Options] FAILED {today} ({', '.join(failed_labels)})",
            combined
        )
        sys.exit(1)

    # Git commit + push
    pushed, git_out = git_push([PARQUET_KC, PARQUET_CC, PARQUET_SB, ATM_JSON])
    log("Git push: OK" if pushed else "Git push: FAILED (may be nothing new)")
    for line in git_out.strip().splitlines():
        log(f"  {line}")

    # Email summary
    body_parts = []
    for label, (ok, out) in all_output.items():
        body_parts.append(f"=== {label} ===\n{out.strip()}")
    body = (
        f"Options ingest completed — {today}\n\n"
        + "\n\n".join(body_parts)
        + f"\n\nGit: {'pushed' if pushed else 'nothing new / failed'}\n{git_out.strip()}"
    )
    send_email(f"[ICEBREAKER Options] OK {today}", body)
    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        msg = traceback.format_exc()
        log(f"UNHANDLED ERROR:\n{msg}")
        send_email(
            f"[ICEBREAKER Options] CRASHED {datetime.date.today()}",
            msg
        )
        sys.exit(1)
