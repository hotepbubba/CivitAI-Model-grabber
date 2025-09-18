"""Graphical interface for the Civitai model downloader."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

from civitAI_Model_downloader import (
    DownloaderConfig,
    VALID_DOWNLOAD_TYPES,
    run_downloader,
)


class DownloaderGUI:
    """Tkinter-based interface for managing downloader configuration and execution."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Civitai Model Downloader")
        self.root.geometry("720x640")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.downloader_thread: Optional[threading.Thread] = None

        self._build_widgets()
        self.root.after(200, self._process_log_queue)

    def _build_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Username input
        username_label = ttk.Label(main_frame, text="Usernames (comma or newline separated)")
        username_label.grid(row=0, column=0, sticky=tk.W)
        usernames_entry = ScrolledText(main_frame, height=3)
        usernames_entry.insert(tk.END, "")
        usernames_entry.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 10))
        self.usernames_entry = usernames_entry

        # Token input
        token_label = ttk.Label(main_frame, text="API Token")
        token_label.grid(row=2, column=0, sticky=tk.W)
        self.token_var = tk.StringVar()
        token_entry = ttk.Entry(main_frame, textvariable=self.token_var, show="*")
        token_entry.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(0, 10))

        # Download type selection
        download_type_label = ttk.Label(main_frame, text="Download Type")
        download_type_label.grid(row=4, column=0, sticky=tk.W)
        self.download_type_var = tk.StringVar(value='All')
        download_type_combo = ttk.Combobox(
            main_frame,
            textvariable=self.download_type_var,
            values=VALID_DOWNLOAD_TYPES,
            state="readonly",
        )
        download_type_combo.grid(row=5, column=0, sticky=tk.W, pady=(0, 10))

        # Output directory selection
        output_dir_label = ttk.Label(main_frame, text="Output Directory")
        output_dir_label.grid(row=6, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="model_downloads")
        output_dir_entry = ttk.Entry(main_frame, textvariable=self.output_dir_var)
        output_dir_entry.grid(row=7, column=0, sticky=tk.EW)
        browse_button = ttk.Button(main_frame, text="Browse", command=self._choose_directory)
        browse_button.grid(row=7, column=1, padx=(6, 0))

        # Numeric options frame
        options_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding=(8, 4))
        options_frame.grid(row=8, column=0, columnspan=3, sticky=tk.EW, pady=(12, 10))
        options_frame.columnconfigure((0, 1, 2), weight=1)

        self.retry_delay_var = tk.IntVar(value=10)
        self.max_tries_var = tk.IntVar(value=3)
        self.max_threads_var = tk.IntVar(value=5)

        self._add_spinbox(options_frame, "Retry Delay (s)", self.retry_delay_var, 0, from_=1, to=120)
        self._add_spinbox(options_frame, "Max Tries", self.max_tries_var, 1, from_=1, to=10)
        self._add_spinbox(options_frame, "Max Threads", self.max_threads_var, 2, from_=1, to=20)

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=3, sticky=tk.EW, pady=(4, 10))
        button_frame.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(button_frame, text="Start Download", command=self._start_download)
        self.start_button.grid(row=0, column=0, sticky=tk.EW)

        # Status label
        self.status_var = tk.StringVar(value="Idle")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=10, column=0, columnspan=3, sticky=tk.EW, pady=(0, 8))

        # Log output
        log_label = ttk.Label(main_frame, text="Log")
        log_label.grid(row=11, column=0, sticky=tk.W)
        self.log_text = ScrolledText(main_frame, height=15, state=tk.DISABLED)
        self.log_text.grid(row=12, column=0, columnspan=3, sticky=tk.NSEW)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(12, weight=1)

    def _add_spinbox(self, frame: ttk.Frame, label: str, variable: tk.IntVar, column: int, **kwargs) -> None:
        lbl = ttk.Label(frame, text=label)
        lbl.grid(row=0, column=column, sticky=tk.W)
        spinbox = ttk.Spinbox(frame, textvariable=variable, width=8, **kwargs)
        spinbox.grid(row=1, column=column, sticky=tk.W, padx=(0, 6))

    def _choose_directory(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def _start_download(self) -> None:
        if self.downloader_thread and self.downloader_thread.is_alive():
            messagebox.showinfo("Download in progress", "Please wait for the current download to finish.")
            return

        usernames = self._parse_usernames(self.usernames_entry.get("1.0", tk.END))
        token = self.token_var.get().strip()
        output_dir = self.output_dir_var.get().strip() or "model_downloads"

        if not usernames:
            messagebox.showerror("Validation Error", "Please provide at least one username.")
            return

        if not token:
            messagebox.showerror("Validation Error", "An API token is required for downloads.")
            return

        try:
            config = DownloaderConfig(
                usernames=usernames,
                retry_delay=int(self.retry_delay_var.get()),
                max_tries=int(self.max_tries_var.get()),
                max_threads=int(self.max_threads_var.get()),
                token=token,
                download_type=self.download_type_var.get(),
                output_dir=output_dir,
            )
        except ValueError as exc:
            messagebox.showerror("Configuration Error", str(exc))
            return

        self.status_var.set("Starting download...")
        self.start_button.config(state=tk.DISABLED)
        self._append_log("Starting download...\n")

        self.downloader_thread = threading.Thread(
            target=self._run_downloader,
            args=(config,),
            daemon=True,
        )
        self.downloader_thread.start()

    def _run_downloader(self, config: DownloaderConfig) -> None:
        try:
            results = run_downloader(config, log_callback=self.log_queue.put)
        except Exception as exc:  # pylint: disable=broad-except
            self.log_queue.put(f"Download failed: {exc}")
            self.root.after(0, lambda: self._on_download_complete(success=False))
            return

        summary_lines = []
        for result in results:
            summary_lines.append(
                f"{result['username']}: downloaded {result['downloaded']} | failed {result['failed']} | skipped {result['skipped']}"
            )
        if summary_lines:
            self.log_queue.put("\n".join(summary_lines))

        self.root.after(0, lambda: self._on_download_complete(success=True))

    def _on_download_complete(self, success: bool) -> None:
        self.start_button.config(state=tk.NORMAL)
        self.status_var.set("Download complete" if success else "Download failed")

    def _process_log_queue(self) -> None:
        while not self.log_queue.empty():
            message = self.log_queue.get()
            self._append_log(message + "\n")
        self.root.after(200, self._process_log_queue)

    def _append_log(self, message: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    @staticmethod
    def _parse_usernames(raw: str) -> list[str]:
        candidates = [part.strip() for part in raw.replace("\n", ",").split(",")]
        return [candidate for candidate in candidates if candidate]


def launch() -> None:
    root = tk.Tk()
    DownloaderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
