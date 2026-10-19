"""
Unified Deepfake Audio Detector — Interactive Launcher

Combines file analysis, folder analysis, and live microphone detection
into one entry point. No manual path typing — uses native OS file/folder
picker dialogs.

The detector uses the automatically loaded calibration profile when one is
available, otherwise it uses the balanced default profile.

Usage:
    python run.py
"""

import os
import sys
import colorama
from colorama import Fore, Style

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

colorama.init(autoreset=True)

from main1 import (
    run_live_microphone,
    run_file_analysis,
    run_folder_analysis,
)
from model_utils import DeepfakeAudioDetector


def pick_file():
    """Opens a native file picker dialog and returns the selected path (or None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print(f"{Fore.RED}[!] tkinter not available. Falling back to manual path entry.{Style.RESET_ALL}")
        path = input("Enter full path to the audio file: ").strip().strip('"')
        return path if path else None

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.wma *.webm"),
            ("All files", "*.*"),
        ]
    )
    root.destroy()
    return file_path if file_path else None


def pick_folder():
    """Opens a native folder picker dialog and returns the selected path (or None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print(f"{Fore.RED}[!] tkinter not available. Falling back to manual path entry.{Style.RESET_ALL}")
        path = input("Enter full path to the folder: ").strip().strip('"')
        return path if path else None

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder_path = filedialog.askdirectory(title="Select a folder of audio files")
    root.destroy()
    return folder_path if folder_path else None


def choose_mode():
    """Interactive menu — returns 'file', 'folder', or 'live'."""
    print("\n" + "=" * 65)
    print("   🎙️  DHWANI DEEPFAKE AUDIO DETECTOR")
    print("=" * 65)
    print("  What would you like to do?\n")
    print("    [1] Analyze a single audio FILE")
    print("    [2] Analyze all audio files in a FOLDER")
    print("    [3] Start LIVE microphone detection")
    print("    [Q] Quit")
    print()

    while True:
        choice = input("Enter your choice (1/2/3/Q): ").strip().lower()
        if choice in ("1", "file", "f"):
            return "file"
        elif choice in ("2", "folder", "d"):
            return "folder"
        elif choice in ("3", "live", "l"):
            return "live"
        elif choice in ("q", "quit", "exit"):
            return None
        else:
            print(f"{Fore.YELLOW}Invalid choice. Please enter 1, 2, 3, or Q.{Style.RESET_ALL}")


def main():
    while True:
        mode = choose_mode()
        if mode is None:
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}\n")
            break

        print(f"\n{Fore.CYAN}[*] Initializing calibrated detector...{Style.RESET_ALL}")
        # Keep the interactive launcher on the same standard decision
        # threshold as main1.py. This is important for multilingual samples
        # whose raw fake score can fall between 0.42 and 0.50.
        detector = DeepfakeAudioDetector(threshold=0.42, verbose=False)

        if mode == "file":
            print(f"\n{Fore.CYAN}[*] Opening file picker...{Style.RESET_ALL}")
            file_path = pick_file()
            if not file_path:
                print(f"{Fore.YELLOW}[!] No file selected.{Style.RESET_ALL}")
            else:
                run_file_analysis(detector, file_path)

        elif mode == "folder":
            print(f"\n{Fore.CYAN}[*] Opening folder picker...{Style.RESET_ALL}")
            folder_path = pick_folder()
            if not folder_path:
                print(f"{Fore.YELLOW}[!] No folder selected.{Style.RESET_ALL}")
            else:
                run_folder_analysis(detector, folder_path)

        elif mode == "live":
            run_live_microphone(detector, device_index=None)

        print()
        again = input(f"{Fore.CYAN}Return to main menu? (y/n): {Style.RESET_ALL}").strip().lower()
        if again not in ("y", "yes", ""):
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}\n")
            break


if __name__ == "__main__":
    main()