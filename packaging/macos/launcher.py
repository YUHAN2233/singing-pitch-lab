"""Local macOS desktop launcher for Tingjian's bundled web application."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import queue
import sys
import socket
import subprocess
import threading
import time
import traceback
from urllib.request import urlopen
import webbrowser


APP_NAME = "听见"
USER_ROOT = Path(os.environ.get("TINGJIAN_USER_ROOT", Path.home() / "Library" / "Application Support" / "Tingjian"))
DATA_ROOT = USER_ROOT / "data"
MODEL_ROOT = USER_ROOT / "models"
SOCKET_PATH = USER_ROOT / "control.sock"
STATE_PATH = USER_ROOT / "state.json"


def send_command(command: str) -> bool:
    for _ in range(40):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(1)
                client.connect(str(SOCKET_PATH))
                client.sendall(command.encode("ascii"))
                return client.recv(2) == b"OK"
        except OSError:
            time.sleep(0.25)
    return False


def free_port() -> int:
    for port in range(8765, 8786):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Ports 8765–8785 are in use.")


def ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
            return json.load(response).get("privacy") == "local-only"
    except Exception:
        return False


def main() -> int:
    USER_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(exist_ok=True)
    MODEL_ROOT.mkdir(exist_ok=True)
    lock_file = (USER_ROOT / "instance.lock").open("a+b")
    try:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0 if send_command("quit" if "--quit" in os.sys.argv[1:] else "open") else 1

        if "--quit" in os.sys.argv[1:]:
            return 0
        SOCKET_PATH.unlink(missing_ok=True)
        os.environ["SINGING_LAB_DATA"] = str(DATA_ROOT)
        os.environ["SINGING_LAB_MODEL"] = str(MODEL_ROOT / "UVR_MDXNET_Main.onnx")
        if getattr(sys, 'frozen', False):
            os.environ['IMAGEIO_FFMPEG_EXE'] = str(Path(sys._MEIPASS) / 'native' / 'ffmpeg')

        import uvicorn
        from backend.app import app, JOBS, LOCK, POOL

        port = free_port()
        url = f"http://127.0.0.1:{port}/"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                                log_level="warning", access_log=False))
        worker = threading.Thread(target=server.run, daemon=True, name="tingjian-server")
        worker.start()
        commands: queue.Queue[str] = queue.Queue()
        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control.bind(str(SOCKET_PATH))
        control.listen(4)
        control.settimeout(0.25)
        stopping = threading.Event()

        def receive() -> None:
            while not stopping.is_set():
                try:
                    client, _ = control.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with client:
                    try:
                        command = client.recv(16).decode("ascii")
                        if command in ("open", "quit"):
                            commands.put(command)
                            client.sendall(b"OK")
                    except OSError:
                        pass

        listener = threading.Thread(target=receive, daemon=True, name="tingjian-control")
        listener.start()
        STATE_PATH.write_text(json.dumps({"pid": os.getpid(), "port": port}), encoding="utf-8")
        launched = False
        closing = False
        closing_at = 0.0

        def close() -> None:
            nonlocal closing, closing_at
            if closing:
                return
            closing = True
            closing_at = time.monotonic()
            with LOCK:
                for job in JOBS.values():
                    job.cancel.set()
            POOL.shutdown(wait=False, cancel_futures=True)
            server.should_exit = True

        def poll() -> None:
            nonlocal launched
            while not commands.empty():
                command = commands.get_nowait()
                if command == "quit":
                    close()
                elif command == "open" and launched and os.environ.get("TINGJIAN_NO_BROWSER") != "1":
                    webbrowser.open(url)
            if not closing and server.started and ready(port) and not launched:
                launched = True
                if os.environ.get("TINGJIAN_NO_BROWSER") != "1":
                    webbrowser.open(url)
            if not closing and not worker.is_alive():
                close()

        try:
            if os.environ.get("TINGJIAN_HEADLESS") == "1":
                while not closing:
                    poll()
                    time.sleep(0.2)
            else:
                import tkinter as tk
                from tkinter import messagebox

                window = tk.Tk()
                window.title(APP_NAME)
                window.geometry("370x165")
                window.resizable(False, False)
                label = tk.StringVar(value="正在启动本地服务…")
                tk.Label(window, text=APP_NAME, font=("Helvetica", 20, "bold")).pack(pady=(12, 3))
                tk.Label(window, textvariable=label).pack(pady=3)
                buttons = tk.Frame(window)
                buttons.pack(pady=9)
                tk.Button(buttons, text="打开页面", command=lambda: webbrowser.open(url)).pack(side="left", padx=4)
                tk.Button(buttons, text="打开模型文件夹",
                          command=lambda: subprocess.Popen(["open", str(MODEL_ROOT)])).pack(side="left", padx=4)
                tk.Button(buttons, text="退出", command=close).pack(side="left", padx=4)
                window.protocol("WM_DELETE_WINDOW", close)

                def tick() -> None:
                    poll()
                    if closing and (not worker.is_alive() or time.monotonic() - closing_at > 15):
                        window.destroy()
                        return
                    label.set(f"本地服务：http://127.0.0.1:{port}" if launched else "正在停止…" if closing else "正在启动本地服务…")
                    window.after(200, tick)

                window.after(200, tick)
                window.mainloop()
                if not closing:
                    close()
            worker.join(timeout=15)
            return 0 if not worker.is_alive() else 1
        finally:
            stopping.set()
            control.close()
            listener.join(timeout=2)
            SOCKET_PATH.unlink(missing_ok=True)
            STATE_PATH.unlink(missing_ok=True)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        USER_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = USER_ROOT / "launcher-error.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        if os.environ.get("TINGJIAN_HEADLESS") != "1":
            try:
                from tkinter import messagebox
                messagebox.showerror(APP_NAME, f"启动失败。详情已写入：\n{log_path}")
            except Exception:
                pass
        raise
