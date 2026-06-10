import logging
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

# Ensure the parent directories are in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_worker_grpc import run_grpc_client

# Log handler to redirect Python logs to the Tkinter text box
class TkinterLogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        # Schedule update on the main Tkinter thread
        self.text_widget.after(0, append)


class AIWorkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceRec AI Worker Node")
        self.root.geometry("680x480")
        self.root.minsize(500, 350)

        # Apply a dark theme styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark color palette
        style.configure(".", background="#1e1e24", foreground="#ffffff", fieldbackground="#2e2e38")
        style.configure("TLabel", background="#1e1e24", foreground="#e0e0e8", font=("Segoe UI", 10))
        style.configure("TButton", background="#3e3e4a", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#4e4e5a")])
        style.configure("Primary.TButton", background="#0066cc", foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#0052a3")])
        style.configure("Danger.TButton", background="#cc3333", foreground="#ffffff")
        style.map("Danger.TButton", background=[("active", "#a32929")])

        self.root.configure(background="#1e1e24")

        # Worker state
        self.worker_thread = None
        self.stop_event = None
        self.is_running = False

        self.setup_ui()
        self.setup_logging()

    def setup_ui(self):
        # Configuration Frame
        config_frame = ttk.LabelFrame(self.root, text=" Configuration ", padding=15)
        config_frame.pack(fill=tk.X, padx=15, pady=10)

        # 1. Control Plane URL
        ttk.Label(config_frame, text="Control Plane URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.url_var = tk.StringVar(value=os.getenv("CONTROL_PLANE_URL", "localhost:50051"))
        self.url_entry = ttk.Entry(config_frame, textvariable=self.url_var, width=35)
        self.url_entry.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)

        # 2. ONNX Execution Provider
        ttk.Label(config_frame, text="Execution Provider:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.provider_var = tk.StringVar(value="Default")
        providers = [
            "Default",
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "ROCmExecutionProvider",
            "DmlExecutionProvider"
        ]
        self.provider_menu = ttk.OptionMenu(config_frame, self.provider_var, "Default", *providers)
        self.provider_menu.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)

        # Action Buttons
        self.start_btn = ttk.Button(config_frame, text="⚡ Start Worker", style="Primary.TButton", command=self.toggle_worker)
        self.start_btn.grid(row=0, column=2, rowspan=2, padx=20, pady=5, sticky=tk.NSEW)

        # Log Frame
        log_frame = ttk.LabelFrame(self.root, text=" Application Logs ", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD, 
            background="#121214", 
            foreground="#a0a0a5", 
            insertbackground="white",
            font=("Consolas", 9.5)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')

    def setup_logging(self):
        # Attach our custom log handler to root logger
        root_logger = logging.getLogger()
        
        # Avoid adding duplicate handlers if re-instantiated
        for handler in root_logger.handlers[:]:
            if isinstance(handler, TkinterLogHandler):
                root_logger.removeHandler(handler)

        handler = TkinterLogHandler(self.log_text)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    def toggle_worker(self):
        if self.is_running:
            self.stop_worker()
        else:
            self.start_worker()

    def start_worker(self):
        url = self.url_var.get().strip()
        provider = self.provider_var.get()
        if provider == "Default":
            provider = None

        if not url:
            logging.error("Control Plane URL cannot be empty.")
            return

        # Disable input fields
        self.url_entry.configure(state='disabled')
        self.provider_menu.configure(state='disabled')
        self.start_btn.configure(text="🛑 Stop Worker", style="Danger.TButton")
        
        self.stop_event = threading.Event()
        self.is_running = True

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self.run_worker_proc,
            args=(url, provider, self.stop_event),
            daemon=True
        )
        self.worker_thread.start()

    def stop_worker(self):
        if self.stop_event:
            logging.info("Stopping AI Worker thread...")
            self.stop_event.set()
        
        self.is_running = False
        self.start_btn.configure(text="⚡ Start Worker", style="Primary.TButton")
        self.url_entry.configure(state='normal')
        self.provider_menu.configure(state='normal')

    def run_worker_proc(self, url, provider, stop_event):
        try:
            run_grpc_client(control_plane_url=url, onnx_provider=provider, stop_event=stop_event)
        except Exception as e:
            logging.error(f"Worker crashed: {e}")
        finally:
            # Update UI on worker thread exit
            def reset_ui():
                if self.is_running:
                    self.stop_worker()
            self.root.after(0, reset_ui)


if __name__ == "__main__":
    # If headless CLI parameters are passed, bypass GUI
    if len(sys.argv) > 1 and ("--help" in sys.argv or "-h" in sys.argv):
        print("Usage: python ai_worker_gui.py [control_plane_url] [onnx_provider]")
        sys.exit(0)

    # Launch GUI
    root = tk.Tk()
    app = AIWorkerGUI(root)
    
    # Handle window close event to clean up thread
    def on_closing():
        if app.is_running:
            app.stop_worker()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
