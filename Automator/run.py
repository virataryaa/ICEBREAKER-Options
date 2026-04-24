"""
run.py -- ICEBREAKER Options daily automator
Runs Ingest.py, commits updated parquet + atm.json, pushes to GitHub, sends email.
"""

import subprocess
import sys
import datetime
import traceback
from pathlib import Path

import win32com.client

# -- Paths --------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parent.parent
INGEST    = ROOT / "Code" / "Ingest.py"
PARQUET   = ROOT / "Database" / "KC_options_ice.parquet"
ATM_JSON  = ROOT / "Dashboard" / "atm.json"
LOG_FILE  = Path(__file__).resolve().parent / "run_log.txt"
PYTHON    = sys.executable

EMAIL_TO  = "virat.arya@etgworld.com"

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


def run_ingest() -> tuple[bool, str]:
    result = subprocess.run(
        [PYTHON, str(INGEST)],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def git_push() -> tuple[bool, str]:
    cmds = [
        ["git", "add",
         str(PARQUET.relative_to(ROOT)),
         str(ATM_JSON.relative_to(ROOT))],
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
    today   = datetime.date.today().isoformat()
    log("=" * 50)
    log(f"KC Options ingest started — {today}")

    # 1. Run ingest
    ok, ingest_out = run_ingest()
    log("Ingest: OK" if ok else "Ingest: FAILED")
    for line in ingest_out.strip().splitlines():
        log("  " + line)

    if not ok:
        send_email(
            f"[ICEBREAKER Options] FAILED {today}",
            f"Ingest failed.\n\n{ingest_out}"
        )
        sys.exit(1)

    # 2. Extract summary line for email
    summary = next(
        (l for l in ingest_out.splitlines() if l.startswith("Saved")),
        ingest_out.splitlines()[-1] if ingest_out.strip() else "No output"
    )

    # 3. Git commit + push
    pushed, git_out = git_push()
    log("Git push: OK" if pushed else "Git push: FAILED (may be nothing new)")
    for line in git_out.strip().splitlines():
        log("  " + line)

    # 4. Email
    body = (
        f"KC Options ingest completed — {today}\n\n"
        f"{ingest_out.strip()}\n\n"
        f"Git: {'pushed' if pushed else 'nothing new / failed'}\n"
        f"{git_out.strip()}"
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
