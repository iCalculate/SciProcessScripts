from __future__ import annotations

import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

if os.name != "nt":
    raise SystemExit("This launcher currently supports Windows only.")

import msvcrt
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist" / "index.html"
DB_ROOT = ROOT / ".mysql"
DB_PORT = 3307
API_PORT = 8010
FRONTEND_PORT = 5173
DB_URL = f"mysql+pymysql://root@127.0.0.1:{DB_PORT}/devicecurvegen"
LOG_LIMIT = 120

console = Console()

APP_MODES = [
    {
        "key": "i",
        "name": "import",
        "label": "Data + Database import mode",
        "detail": "Import panel and Database panel for incremental source ingestion",
        "processes": {"db", "api"},
    },
    {
        "key": "a",
        "name": "analysis",
        "label": "Data + Database + Analysis mode",
        "detail": "Import panel, Database selection, and Analysis workspace",
        "processes": {"db", "api"},
    },
    {
        "key": "g",
        "name": "generate",
        "label": "Pure Generate mode",
        "detail": "Generation-only workbench without database or model tabs",
        "processes": {"api"},
    },
    {
        "key": "t",
        "name": "training",
        "label": "Database + Models training mode",
        "detail": "Database workspace and model training / retraining workspace",
        "processes": {"db", "api"},
    },
    {
        "key": "f",
        "name": "full",
        "label": "Full mode",
        "detail": "All tabs plus Vite dev frontend on port 5173",
        "processes": {"db", "api", "frontend"},
    },
]

PROCESS_META = {
    "db": {"label": "DB", "color": "cyan"},
    "api": {"label": "API", "color": "green"},
    "frontend": {"label": "WEB", "color": "magenta"},
}

URL_PATTERN = re.compile(r"https?://[^\s]+", flags=re.IGNORECASE)
HOST_PORT_PATTERN = re.compile(r"\b(?:127\.0\.0\.1|localhost):\d+\b", flags=re.IGNORECASE)
PORT_PATTERN = re.compile(r"\bport\s+\d+\b", flags=re.IGNORECASE)
ERROR_PATTERN = re.compile(
    r"\b(error|failed|failure|fatal|exception|aborting|denied|offline|traceback)\b",
    flags=re.IGNORECASE,
)
WARN_PATTERN = re.compile(
    r"\b(warn|warning|retry|skipped|timeout)\b",
    flags=re.IGNORECASE,
)
OK_PATTERN = re.compile(
    r"\b(started|running|ready|online|complete|completed|success|successful|listening|alive|imported)\b",
    flags=re.IGNORECASE,
)


def clear_screen() -> None:
    os.system("cls")


def read_key() -> str:
    first = msvcrt.getwch()
    if first in {"\x00", "\xe0"}:
        second = msvcrt.getwch()
        return f"SPECIAL:{ord(second)}"
    return first


def render_menu(cursor: int, message: str | None = None) -> None:
    clear_screen()
    title = Text("FET-GEN Launcher", style="bold bright_cyan")
    console.print(Panel(title, border_style="bright_blue"))
    console.print("[dim]Arrow keys move, Enter launches, Q quits.[/dim]")
    console.print("[dim]Shortcuts: I = import, A = analysis, G = generate, T = training, F = full.[/dim]")
    console.print()
    for index, mode in enumerate(APP_MODES):
        marker = "[bold yellow]>[/bold yellow]" if index == cursor else " "
        console.print(f"{marker} [bold]{mode['label']}[/bold]")
        console.print(f"      [dim]{mode['detail']}[/dim]")
    console.print()
    selected_mode = APP_MODES[cursor]
    console.print(Panel(f"[bold]Selected mode:[/bold] {selected_mode['label']}", border_style="blue"))
    if message:
        console.print(Panel(message, border_style="red", title="Notice"))


def describe_mode(mode_name: str) -> str:
    for mode in APP_MODES:
        if mode["name"] == mode_name:
            return str(mode["label"])
    return mode_name


def stylize_matches(text: Text, pattern: re.Pattern[str], style: str) -> None:
    plain = text.plain
    for match in pattern.finditer(plain):
        text.stylize(style, match.start(), match.end())


def stylize_log_message(message: str) -> Text:
    text = Text(message.rstrip(), style="white")
    stylize_matches(text, URL_PATTERN, "bold underline bright_blue")
    stylize_matches(text, HOST_PORT_PATTERN, "bold bright_cyan")
    stylize_matches(text, PORT_PATTERN, "bold bright_yellow")
    stylize_matches(text, ERROR_PATTERN, "bold bright_red")
    stylize_matches(text, WARN_PATTERN, "bold yellow")
    stylize_matches(text, OK_PATTERN, "bold bright_green")
    return text


def ensure_frontend_build() -> None:
    if FRONTEND_DIST.exists():
        return
    console.print("[yellow]Frontend build is missing. Building it now...[/yellow]")
    corepack = find_corepack_binary()
    if not corepack:
        raise SystemExit("Could not find corepack. Install Node.js with Corepack enabled.")
    run_checked([corepack, "pnpm", "install", "--frozen-lockfile"], cwd=FRONTEND_DIR)
    run_checked([corepack, "pnpm", "build"], cwd=FRONTEND_DIR)


def run_checked(command: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["COREPACK_ENABLE_AUTO_PIN"] = "0"
    completed = subprocess.run(command, cwd=cwd, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def find_mariadb_binary() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\MariaDB 12.3\bin\mariadbd.exe"),
        Path(r"C:\Program Files\MariaDB 11.8\bin\mariadbd.exe"),
        Path(r"C:\Program Files\MariaDB 11.7\bin\mariadbd.exe"),
        Path(r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqld.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    which = shutil.which("mariadbd.exe") or shutil.which("mysqld.exe")
    return Path(which) if which else None


def find_corepack_binary() -> str | None:
    candidates = [
        shutil.which("corepack.cmd"),
        shutil.which("corepack.exe"),
        shutil.which("corepack"),
        r"C:\Program Files\nodejs\corepack.cmd",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_open(port):
            return True
        time.sleep(0.25)
    return False


def stop_listener(port: int) -> None:
    subprocess.run(
        ["cmd.exe", "/c", f'for /f "tokens=5" %P in (\'netstat -ano ^| findstr /R /C:":{port} .*LISTENING"\') do taskkill /PID %P /F'],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def maybe_stop_existing(selected: set[str]) -> None:
    ports: list[int] = []
    if "db" in selected:
        ports.append(DB_PORT)
    if "api" in selected:
        ports.append(API_PORT)
    if "frontend" in selected:
        ports.append(FRONTEND_PORT)
    occupied = [port for port in ports if port_is_open(port)]
    if not occupied:
        return
    clear_screen()
    console.print(Panel(
        "The following project ports are already in use:\n"
        + ", ".join(str(port) for port in occupied)
        + "\n\nStop the existing listeners and continue? [Y/n]",
        border_style="yellow",
        title="Ports Busy",
    ))
    key = read_key().lower()
    if key == "n":
        raise SystemExit(0)
    for port in occupied:
        stop_listener(port)
    time.sleep(2)


def prepare_db_permissions() -> None:
    db_user = f"{os.environ['COMPUTERNAME']}\\{os.environ['USERNAME']}"
    subprocess.run(
        ["cmd.exe", "/c", f'attrib -R "{DB_ROOT}\\*" /S /D'],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["takeown", "/F", str(DB_ROOT), "/R", "/D", "Y"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["icacls", str(DB_ROOT), "/grant", f"{db_user}:(OI)(CI)F", "/T", "/C"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def spawn_process(name: str, command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    popen_env = os.environ.copy()
    if env:
        popen_env.update(env)
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=popen_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


def start_db() -> subprocess.Popen[str]:
    binary = find_mariadb_binary()
    if not binary:
        raise SystemExit("Could not find mariadbd.exe or mysqld.exe.")
    prepare_db_permissions()
    return spawn_process(
        "db",
        [
            str(binary),
            f"--defaults-file={DB_ROOT / 'my.ini'}",
            f"--port={DB_PORT}",
            "--skip-grant-tables",
            "--ssl=0",
            "--console",
        ],
        cwd=DB_ROOT,
    )


def start_api(app_mode: str) -> subprocess.Popen[str]:
    return spawn_process(
        "api",
        [str(VENV_PYTHON), "-m", "devicecurvegen.cli", "serve", "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=ROOT,
        env={"DEVICEGEN_DATABASE_URL": DB_URL, "DEVICEGEN_APP_MODE": app_mode},
    )


def start_frontend() -> subprocess.Popen[str]:
    corepack = find_corepack_binary()
    if not corepack:
        raise SystemExit("Could not find corepack. Install Node.js with Corepack enabled.")
    return spawn_process(
        "frontend",
        [corepack, "pnpm", "dev", "--host", "127.0.0.1", "--port", str(FRONTEND_PORT)],
        cwd=FRONTEND_DIR,
        env={"COREPACK_ENABLE_AUTO_PIN": "0"},
    )


def format_log_line(source: str, line: str) -> Text:
    meta = PROCESS_META[source]
    prefix = Text.assemble(
        ("[", "dim"),
        (f"{meta['label']:^3}", f"bold {meta['color']}"),
        ("] ", "dim"),
    )
    message = stylize_log_message(line)
    return prefix + message


def drain_process_output(
    source: str,
    process: subprocess.Popen[str],
    output_queue: queue.Queue[tuple[str, str]],
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        output_queue.put((source, line.rstrip()))
    process.stdout.close()


def print_runtime_header(mode: dict[str, object], selected: set[str]) -> None:
    clear_screen()
    console.print(Panel(Text("FET-GEN Running", style="bold bright_green"), border_style="bright_green"))
    console.print(Text.assemble(("Mode", "bold white"), (": ", "dim"), (str(mode["label"]), "bold bright_white")))
    if "db" in selected:
        console.print(Text.assemble(
            ("DB", "bold cyan"),
            ("  ", "white"),
            ("127.0.0.1", "bright_cyan"),
            (":", "dim"),
            (str(DB_PORT), "bold bright_yellow"),
            ("  local database", "dim"),
        ))
    if "api" in selected:
        console.print(Text.assemble(
            ("API", "bold green"),
            (" ", "white"),
            (f"http://127.0.0.1:{API_PORT}", "bold underline bright_blue"),
            ("  analyzer backend", "dim"),
        ))
    if "frontend" in selected:
        console.print(Text.assemble(
            ("WEB", "bold magenta"),
            (" ", "white"),
            (f"http://127.0.0.1:{FRONTEND_PORT}", "bold underline bright_blue"),
            ("  frontend dev server", "dim"),
        ))
    elif "api" in selected:
        console.print(Text.assemble(
            ("WEB", "bold magenta"),
            (" ", "white"),
            (f"http://127.0.0.1:{API_PORT}", "bold underline bright_blue"),
            (f"  built app ({mode['name']})", "dim"),
        ))
    console.print("[dim]Press Q to stop all launched processes.[/dim]")
    console.print()


def monitor_processes(mode: dict[str, object], selected: set[str], processes: dict[str, subprocess.Popen[str]]) -> None:
    output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    log_buffer: deque[Text] = deque(maxlen=LOG_LIMIT)
    threads: list[threading.Thread] = []

    for source, process in processes.items():
        thread = threading.Thread(
            target=drain_process_output,
            args=(source, process, output_queue),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    print_runtime_header(mode, selected)
    while True:
        while True:
            try:
                source, line = output_queue.get_nowait()
            except queue.Empty:
                break
            entry = format_log_line(source, line)
            log_buffer.append(entry)
            console.print(entry)

        dead = [name for name, process in processes.items() if process.poll() is not None]
        if dead:
            for name in dead:
                console.print(
                    Panel(
                        Text.assemble(
                            (PROCESS_META[name]["label"], f"bold {PROCESS_META[name]['color']}"),
                            (" exited with code ", "white"),
                            (str(processes[name].returncode), "bold bright_yellow"),
                            (".", "white"),
                        ),
                        border_style="red",
                        title="Process Stopped",
                    )
                )
            break

        if msvcrt.kbhit():
            key = read_key()
            if key.lower() == "q":
                console.print("[yellow]Stopping launched processes...[/yellow]")
                break

        time.sleep(0.1)

    stop_processes(processes)
    for thread in threads:
        thread.join(timeout=1)
    console.print("[bold]Launcher finished.[/bold]")
    console.print("[dim]Press any key to close this window.[/dim]")
    msvcrt.getwch()


def stop_processes(processes: dict[str, subprocess.Popen[str]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 5
    while time.time() < deadline:
        if all(process.poll() is not None for process in processes.values()):
            return
        time.sleep(0.2)
    for process in processes.values():
        if process.poll() is None:
            process.kill()


def launch(mode: dict[str, object]) -> None:
    selected = set(mode["processes"])
    maybe_stop_existing(selected)
    if "api" in selected and "frontend" not in selected:
        ensure_frontend_build()

    processes: dict[str, subprocess.Popen[str]] = {}
    try:
        if "db" in selected:
            processes["db"] = start_db()
            if not wait_for_port(DB_PORT, 12):
                raise SystemExit("Local database failed to start on port 3307.")
        if "api" in selected:
            processes["api"] = start_api(str(mode["name"]))
            if not wait_for_port(API_PORT, 12):
                raise SystemExit("Analyzer API failed to start on port 8010.")
        if "frontend" in selected:
            processes["frontend"] = start_frontend()
            if not wait_for_port(FRONTEND_PORT, 12):
                raise SystemExit("Frontend dev server failed to start on port 5173.")
        monitor_processes(mode, selected, processes)
    except Exception:
        stop_processes(processes)
        raise


def main() -> int:
    if not VENV_PYTHON.exists():
        console.print("[red]Local environment is missing. Run setup-local-env.bat first.[/red]")
        return 1

    cursor = 1
    message: str | None = None

    while True:
        render_menu(cursor, message)
        message = None
        key = read_key()
        if key.lower() == "q":
            return 0
        if key in {"\r", "\n"}:
            mode = APP_MODES[cursor]
            try:
                launch(mode)
                return 0
            except SystemExit as exc:
                message = str(exc)
                continue
        mode_index = next((index for index, mode in enumerate(APP_MODES) if mode["key"] == key.lower()), None)
        if mode_index is not None:
            cursor = mode_index
            continue
        if key == "SPECIAL:72":
            cursor = (cursor - 1) % len(APP_MODES)
            continue
        if key == "SPECIAL:80":
            cursor = (cursor + 1) % len(APP_MODES)
            continue


if __name__ == "__main__":
    raise SystemExit(main())
