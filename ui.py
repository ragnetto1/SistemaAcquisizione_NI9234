# Data/Ora: 2026-02-12 20:38:51
# ui.py
from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import configparser
import re
import shutil  # per rimuovere cartelle temporanee dopo merge
import pyqtgraph as pg
from collections import deque
import numpy as np
import os
import xml.etree.ElementTree as ET
import datetime
import json
import glob
import importlib.util
import time
from typing import List, Callable, Optional, Dict, Any, Tuple

try:
    from syncronizzation import ModuleSyncAgent
except Exception:
    ModuleSyncAgent = None

# Path to store persistent configuration. It resides alongside this script.
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

#
# Disable OpenGL rendering for pyqtgraph.  Using OpenGL together with
# PlotCurveItem.setData can lead to a memory leak on some systems.  See
# https://github.com/pyqtgraph/pyqtgraph/issues/3372 for discussion.  By
# disabling OpenGL here we force pyqtgraph to use its native Qt painting
# backend which has much more predictable memory usage.  Antialiasing is
# also disabled to keep CPU usage modest.
pg.setConfigOptions(useOpenGL=False, antialias=False)

# Colonne tabella
COL_ENABLE   = 0
COL_PHYS     = 1
COL_TYPE     = 2   # Tipo risorsa (Voltage o sensori dal DB)
COL_LABEL    = 3   # Nome canale (etichetta utente)
COL_VALUE    = 4   # Valore istantaneo (con unit? se selezionata)
COL_ZERO_BTN = 5
COL_ZERO_VAL = 6
# New columns for NI-9234 coupling and sensor limits.
COL_COUPLING = 7
COL_LIMIT_MAX = 8
COL_LIMIT_MIN = 9

# Percorso di default richiesto
# Per la NI-9234 il progetto utilizza directory dedicate.  I percorsi
# predefiniti possono essere personalizzati modificando queste costanti.
DEFAULT_SAVE_DIR = r"C:\UG-WORK\SistemaAcquisizione_NI9234"
SENSOR_DB_DEFAULT = r"C:\UG-WORK\SistemaAcquisizione_NI9234\Sensor database.xml"

# XML tag (compat vecchio e nuovo formato multi-punti)
XML_ROOT  = "Sensors"
XML_ITEM  = "Sensor"
XML_NAME  = "NomeRisorsa"
XML_UNIT  = "GrandezzaFisica"
XML_CAL   = "CalibrationPoints"
XML_POINT = "Point"          # attr: volt, value
# vecchio (2 punti)
XML_V1V = "Valore1Volt"
XML_V1  = "Valore1"
XML_V2V = "Valore2Volt"
XML_V2  = "Valore2"


class FFTDiskWorker(QtCore.QObject):
    sigStatus = QtCore.pyqtSignal(str)
    sigResult = QtCore.pyqtSignal(object, object, object)   # (freq, mag_by_phys, peak_text)
    sigGuardrail = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._duration_s = 5.0
        self._save_dir = DEFAULT_SAVE_DIR
        self._phys_order: List[str] = []
        self._chunk_dir = ""
        self._window_idx = 0
        self._chunk_samples = 0
        self._chunk_fs_est = 0.0
        self._window_start_t = None
        self._window_end_t = None
        self._window_t_path = ""
        self._window_y_paths: Dict[str, str] = {}
        self._temp_npz_files: List[str] = []
        self._temp_keep_count = 3

    def _get_chunk_dir(self) -> str:
        base = self._save_dir or DEFAULT_SAVE_DIR
        return os.path.join(base, "_fft_chunks")

    def _ensure_chunk_dir(self) -> bool:
        d = self._get_chunk_dir()
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            return False
        self._chunk_dir = d
        return True

    def _open_new_window_files(self) -> bool:
        if not self._ensure_chunk_dir():
            return False
        win = int(self._window_idx)
        self._window_t_path = os.path.join(self._chunk_dir, f"fft_window_{win:06d}_t.f64.bin")
        self._window_y_paths = {
            phys: os.path.join(self._chunk_dir, f"fft_window_{win:06d}_{phys}.f32.bin")
            for phys in self._phys_order
        }
        return True

    def _estimate_fs(self, t: np.ndarray) -> float:
        if not isinstance(t, np.ndarray) or t.size < 2:
            return 0.0
        try:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0.0)]
            if dt.size < 1:
                return 0.0
            fs = 1.0 / float(np.median(dt))
            return fs if np.isfinite(fs) and fs > 0.0 else 0.0
        except Exception:
            return 0.0

    def _compute_fft_mag(self, x: np.ndarray, fs: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(x, np.ndarray):
            return None, None
        n = int(x.size)
        if n < 64 or fs <= 0.0:
            return None, None
        nfft = 1 << int(np.floor(np.log2(max(64, n))))
        if nfft < 64:
            return None, None
        seg = np.asarray(x[-nfft:], dtype=np.float64, copy=False)
        seg = seg - float(np.mean(seg))
        win = np.hanning(nfft).astype(np.float64, copy=False)
        w_sum = float(np.sum(win))
        if w_sum <= 0.0:
            return None, None
        spec = np.fft.rfft(seg * win)
        mag = (2.0 / w_sum) * np.abs(spec)
        freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
        return freq, mag

    def _trim_temp_npz(self) -> None:
        keep_n = max(1, int(self._temp_keep_count))
        while len(self._temp_npz_files) > keep_n:
            old = self._temp_npz_files.pop(0)
            try:
                if os.path.isfile(old):
                    os.remove(old)
            except Exception:
                pass

    def _should_abort(self) -> bool:
        try:
            th = QtCore.QThread.currentThread()
            if th is not None and th.isInterruptionRequested():
                return True
        except Exception:
            pass
        return False

    @QtCore.pyqtSlot(object)
    def configure(self, cfg: object) -> None:
        if not isinstance(cfg, dict):
            return
        try:
            self._duration_s = max(0.1, float(cfg.get("duration_s", self._duration_s)))
        except Exception:
            pass
        try:
            self._save_dir = str(cfg.get("save_dir", self._save_dir) or self._save_dir)
        except Exception:
            pass
        try:
            po = cfg.get("phys_order", self._phys_order)
            if isinstance(po, (list, tuple)):
                self._phys_order = [str(x) for x in po]
        except Exception:
            pass
        try:
            self._temp_keep_count = max(1, int(cfg.get("temp_keep_count", self._temp_keep_count)))
        except Exception:
            pass

        # Guardrail warning only; no decimation.
        try:
            fs = float(cfg.get("fs_hz", 0.0) or 0.0)
            ch = max(1, len(self._phys_order))
            total = fs * self._duration_s * ch
            if total >= 1_000_000:
                self.sigGuardrail.emit(
                    f"Guardrail FFT: finestra molto pesante ({total:.0f} punti totali)."
                )
        except Exception:
            pass

    @QtCore.pyqtSlot(bool, bool)
    def reset(self, cleanup_files: bool, cleanup_temp_npz: bool) -> None:
        if cleanup_files:
            d = self._chunk_dir or self._get_chunk_dir()
            if d and os.path.isdir(d):
                try:
                    for p in glob.glob(os.path.join(d, "fft_window_*")):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                    if cleanup_temp_npz:
                        for p in glob.glob(os.path.join(d, "fft_temp_window_*.npz")):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                        try:
                            shutil.rmtree(d, ignore_errors=True)
                        except Exception:
                            pass
                except Exception:
                    pass
        self._chunk_samples = 0
        self._chunk_fs_est = 0.0
        self._window_start_t = None
        self._window_end_t = None
        self._window_t_path = ""
        self._window_y_paths = {}
        if cleanup_files and cleanup_temp_npz:
            self._chunk_dir = ""
            self._temp_npz_files = []

    @QtCore.pyqtSlot(object)
    def process_block(self, payload: object) -> None:
        if self._should_abort():
            return
        if not isinstance(payload, dict):
            return
        try:
            t = np.asarray(payload.get("t", []), dtype=np.float64)
        except Exception:
            return
        if t.size < 2:
            return
        y_map = payload.get("y_map", {})
        if not isinstance(y_map, dict):
            return
        if not self._window_t_path:
            if not self._open_new_window_files():
                return

        fs_est = self._estimate_fs(t)
        if fs_est > 0.0:
            self._chunk_fs_est = fs_est

        try:
            with open(self._window_t_path, "ab") as f:
                t.tofile(f)
        except Exception:
            return

        for phys in self._phys_order:
            y = y_map.get(phys)
            if isinstance(y, np.ndarray) and y.size == t.size:
                y_arr = np.asarray(y, dtype=np.float32)
            else:
                y_arr = np.full(int(t.size), np.nan, dtype=np.float32)
            p = self._window_y_paths.get(phys, "")
            if not p:
                continue
            try:
                with open(p, "ab") as f:
                    y_arr.tofile(f)
            except Exception:
                pass

        self._chunk_samples += int(t.size)
        t0 = float(t[0]); t1 = float(t[-1])
        if self._window_start_t is None:
            self._window_start_t = t0
        self._window_end_t = t1

        if self._window_start_t is None or self._window_end_t is None:
            return
        elapsed = max(0.0, float(self._window_end_t) - float(self._window_start_t))
        target = max(0.1, float(self._duration_s))
        fmax = max(0.0, float(self._chunk_fs_est or 0.0) / 2.0)
        self.sigStatus.emit(f"FFT: finestra su disco {elapsed:.2f}/{target:g}s - F max= {fmax:.3g} Hz")
        if elapsed < target:
            return

        if self._should_abort():
            return
        if not self._window_t_path or not os.path.isfile(self._window_t_path):
            self.reset(True, False)
            self._window_idx += 1
            return
        t_all = np.fromfile(self._window_t_path, dtype=np.float64)
        if t_all.size < 64:
            self.reset(True, False)
            self._window_idx += 1
            return

        y_all = {}
        n = int(t_all.size)
        for phys in self._phys_order:
            p = self._window_y_paths.get(phys, "")
            if p and os.path.isfile(p):
                y = np.fromfile(p, dtype=np.float32)
                if y.size < n:
                    y = np.concatenate([y, np.full(n - y.size, np.nan, dtype=np.float32)])
                elif y.size > n:
                    y = y[:n]
            else:
                y = np.full(n, np.nan, dtype=np.float32)
            y_all[phys] = y

        final_npz = os.path.join(self._chunk_dir, f"fft_temp_window_{int(self._window_idx):06d}.npz")
        out = {"t": t_all, "phys": np.asarray(self._phys_order, dtype=object)}
        for phys in self._phys_order:
            out[f"y_{phys}"] = y_all[phys]
        np.savez(final_npz, **out)
        self._temp_npz_files.append(final_npz)
        self._trim_temp_npz()

        fs = self._estimate_fs(t_all)
        if fs <= 0.0:
            fs = float(self._chunk_fs_est or 0.0)
        if fs <= 0.0:
            self.reset(True, False)
            self._window_idx += 1
            return

        freq_ref = None
        mag_map: Dict[str, np.ndarray] = {}
        peaks = []
        for phys in self._phys_order:
            y = np.asarray(y_all.get(phys, []), dtype=np.float64)
            if y.size != t_all.size or y.size < 64:
                continue
            if not np.all(np.isfinite(y)):
                v = np.isfinite(y)
                if np.count_nonzero(v) < 64:
                    continue
                y = np.interp(t_all, t_all[v], y[v])
            freq, mag = self._compute_fft_mag(y, fs)
            if not (isinstance(freq, np.ndarray) and isinstance(mag, np.ndarray)):
                continue
            if freq_ref is None:
                freq_ref = freq
            if freq_ref is None or mag.size != freq_ref.size:
                continue
            mag_map[phys] = mag
            if mag.size > 1:
                i = int(np.argmax(mag[1:])) + 1
                peaks.append((phys, float(mag[i]), float(freq_ref[i])))

        peak_text = "; ".join([f"{p}: {a:.3g} @ {f:.6g} Hz" for p, a, f in peaks])
        if isinstance(freq_ref, np.ndarray) and mag_map:
            self.sigResult.emit(freq_ref, mag_map, peak_text)

        self.reset(True, False)
        self._window_idx += 1


class AcquisitionWindow(QtWidgets.QMainWindow):
    # segnali thread-safe verso UI
    sigInstantBlock = QtCore.pyqtSignal(object, object, object)   # (t, [ys...], [names...])
    sigChartPoints  = QtCore.pyqtSignal(object, object, object)
    channelValueUpdated = QtCore.pyqtSignal(str, float)           # (start_label_name, value)
    sigFftWorkerConfig = QtCore.pyqtSignal(object)
    sigFftWorkerReset = QtCore.pyqtSignal(bool, bool)
    sigFftWorkerBlock = QtCore.pyqtSignal(object)

    def __init__(self, acq_manager, merger, parent=None):
        super().__init__(parent)
        self.acq = acq_manager
        self.merger = merger

        # Finestra per il modulo NI-9234.  In questo progetto non ? prevista
        # la selezione di altri modelli di scheda.
        self.setWindowTitle("NI 9234 Acquisition - Demo Architettura")
        self.resize(1200, 740)

        # stati UI/logica
        self._building_table = False
        self._auto_change = False
        self._device_ready = False

        # mappature canali
        self._current_phys_order = []                    # ordine fisico corrente avviato
        # La NI-9234 ha quattro canali simultanei
        try:
            num_chans = int(getattr(self.acq, "num_channels", 4))
        except Exception:
            num_chans = 4
        # Inizializza le strutture mappatura e calibrazione per ciascun canale fisico
        self._label_by_phys = {f"ai{i}": f"ai{i}" for i in range(num_chans)}   # label utente "Nome canaleâ€
        self._sensor_type_by_phys = {f"ai{i}": "Voltage" for i in range(num_chans)}
        self._calib_by_phys = {f"ai{i}": {"unit":"", "a":1.0, "b":0.0} for i in range(num_chans)}
        self._start_label_by_phys = {}                   # mapping phys -> nome al momento dello start
        self._last_enabled_phys = []

        # grafici: buffer
        MAXPTS = 12000
        self._chart_x = deque(maxlen=MAXPTS)
        # Buffer e curve grafici per ciascun canale
        self._chart_y_by_phys = {f"ai{i}": deque(maxlen=MAXPTS) for i in range(num_chans)}
        self._instant_t = np.array([], dtype=float)
        self._instant_y_by_phys = {f"ai{i}": np.array([], dtype=float) for i in range(num_chans)}
        self._chart_curves_by_phys = {}
        self._instant_curves_by_phys = {}

        # Initialize dictionary for FFT curves. Each physical channel will
        # have its own curve in the FFT plot. These curves are created when
        # a new acquisition starts in _reset_plots_curves().
        self._fft_curves_by_phys = {}

        # --- FFT configuration ---
        # Duration of the time window (in seconds) used for FFT computation.
        # The FFT is computed from a dedicated low-rate buffer fed by the
        # instant block stream, not from decimated chart points.
        self._fft_duration_seconds: float = 5.0
        # Requested sampling rate for the FFT buffer. The effective target is
        # adapted to duration and memory limits.
        self._fft_target_fs_hz: float = 1000.0
        # Hard cap on FFT buffered samples (all channels share the same time axis).
        self._fft_max_samples: int = 200_000
        # Recompute only when a full new FFT window is available.
        self._fft_update_full_window: bool = True
        self._last_fft_compute_ts: float = 0.0
        self._fft_last_window_end_t: Optional[float] = None
        self._fft_t = deque(maxlen=1)
        self._fft_y_by_phys = {f"ai{i}": deque(maxlen=1) for i in range(num_chans)}
        # Disk-backed FFT pipeline state.
        self._fft_chunk_dir: str = ""
        self._fft_chunk_samples: int = 0
        self._fft_chunk_window_idx: int = 0
        self._fft_chunk_fs_est: float = 0.0
        self._fft_window_start_t: Optional[float] = None
        self._fft_window_end_t: Optional[float] = None
        self._fft_window_t_path: str = ""
        self._fft_window_y_paths: Dict[str, str] = {}
        self._fft_temp_npz_files: List[str] = []
        self._fft_temp_keep_count: int = 3
        # Storage for the last computed FFT frequency vector and magnitude
        # spectra.  These arrays are used both for display and for saving
        # into the TDMS file at merge time.  They are updated in
        # _refresh_plots().
        self._last_fft_freq: Optional[np.ndarray] = None
        self._last_fft_mag_by_phys: Dict[str, np.ndarray] = {}
        # Monotonic counters used to ensure only the latest FFT computed during
        # the current recording window is eligible for TDMS export.
        self._fft_result_counter: int = 0
        self._fft_result_counter_at_record_start: int = 0

        # Flag used internally to indicate that the FFT enable checkbox is
        # being changed programmatically, preventing reentry in the
        # associated slot.
        self._auto_fft_change: bool = False
        self._instant_plot_enabled: bool = False

        # Whether FFT computation is enabled.  When true, FFT is computed in
        # sliding-window mode from the dedicated FFT buffer.
        self._fft_enabled: bool = False
        self._reset_fft_buffers()

        # Directory di salvataggio per la NI-9234 (nessun cambio dinamico per altri modelli)
        self._save_dir = DEFAULT_SAVE_DIR
        self._base_filename = "SenzaNome.tdms"
        self._active_subdir = None
        self._countdown = 60
        self._count_timer = QtCore.QTimer(self)
        self._count_timer.setInterval(1000)
        self._count_timer.timeout.connect(self._tick_countdown)

        # Timer to monitor disk stall/backlog
        self._backlog_timer = QtCore.QTimer(self)
        self._backlog_timer.setInterval(1000)  # check every second
        self._backlog_timer.timeout.connect(self._check_backlog)
        # Default update interval for charts; used to restore after stall.
        # A longer refresh interval (e.g. 100 ms) reduces CPU usage and memory
        # churn associated with converting deques to arrays at high rates.  The
        # memory footprint of charts grows if arrays are reallocated too often.
        self._default_gui_interval = 100
        # Track if we are in stall mode to avoid repeated adjustments
        self._stall_active = False

        # Percorso del database sensori per la NI-9234
        self._sensor_db_path = SENSOR_DB_DEFAULT

        # Controllo remoto da chassis (attivo solo se lanciato dal root con env sync).
        self._sync_agent = None
        self._sync_remote_active = False
        self._sync_arm_requested = False
        self._pending_sync_start_cfg: Optional[Dict[str, Any]] = None

        # UI
        self._build_ui()
        self._init_fft_worker()
        self._connect_signals()

        # Load persistent configuration (if available)
        try:
            self._load_config()
        except Exception:
            pass

        # inizializzazione
        self.refresh_devices()

        # Start backlog monitoring timer
        self._backlog_timer.start()

        # Avvia l'agent solo in modalita chassis-control.
        self._init_sync_agent()

    def _init_fft_worker(self):
        self._fft_thread = QtCore.QThread(self)
        worker_cls = FFTDiskWorker
        try:
            fft_path = os.path.join(os.path.dirname(__file__), "FFT calculation.py")
            if os.path.isfile(fft_path):
                spec = importlib.util.spec_from_file_location("fft_calculation_worker", fft_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    worker_cls = getattr(mod, "FFTDiskWorker", FFTDiskWorker)
        except Exception:
            worker_cls = FFTDiskWorker
        self._fft_worker = worker_cls()
        self._fft_worker.moveToThread(self._fft_thread)
        self.sigFftWorkerConfig.connect(self._fft_worker.configure, QtCore.Qt.QueuedConnection)
        self.sigFftWorkerReset.connect(self._fft_worker.reset, QtCore.Qt.QueuedConnection)
        self.sigFftWorkerBlock.connect(self._fft_worker.process_block, QtCore.Qt.QueuedConnection)
        self._fft_worker.sigStatus.connect(self._on_fft_worker_status)
        self._fft_worker.sigResult.connect(self._on_fft_worker_result)
        self._fft_worker.sigGuardrail.connect(self._on_fft_worker_guardrail)
        try:
            self._fft_thread.finished.connect(self._fft_worker.deleteLater)
        except Exception:
            pass
        self._fft_thread.start()

    # ----------------------------- Build UI -----------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        # Riga superiore: rileva + device + frequenza campionamento + definisci risorse
        top = QtWidgets.QHBoxLayout()
        # Pulsante per rilevare le schede NI presenti
        self.btnRefresh = QtWidgets.QPushButton("Rileva schede")
        top.addWidget(self.btnRefresh)
        # Etichetta e combobox per il dispositivo NI
        top.addWidget(QtWidgets.QLabel("Dispositivo:"))
        self.cmbDevice = QtWidgets.QComboBox()
        top.addWidget(self.cmbDevice, 1)
        # Campo di input per la frequenza di campionamento per canale
        top.addWidget(QtWidgets.QLabel("Fs [Hz]:"))
        self.rateEdit = QtWidgets.QLineEdit()
        # Imposta dimensione fissa per il campo del rate
        self.rateEdit.setFixedWidth(80)
        # Se non impostato, mostra "Max" come suggerimento
        self.rateEdit.setPlaceholderText("Max")
        top.addWidget(self.rateEdit)
        # Pulsante per definire i sensori/risorse
        self.btnDefineTypes = QtWidgets.QPushButton("Definisci Tipo Risorsa")
        top.addWidget(self.btnDefineTypes)
        # Pulsanti per salvare e caricare workspace
        self.btnSaveWorkspace = QtWidgets.QPushButton("Salva workspace")
        self.btnLoadWorkspace = QtWidgets.QPushButton("Carica workspace")
        top.addWidget(self.btnSaveWorkspace)
        top.addWidget(self.btnLoadWorkspace)
        # Allineamento a destra per riempire lo spazio residuo
        main.addLayout(top)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs, 1)

        # Tab Canali: tabella
        tabTable = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tabTable)
        # Create the table with additional columns for Coupling and limits.
        self.table = QtWidgets.QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "Abilita", "Canale fisico", "Tipo risorsa", "Nome canale",
            "Valore istantaneo", "Azzeramento", "Valore azzerato",
            "Coupling", "Limite Max input", "Limite Min input"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)
        self.tabs.addTab(tabTable, "Canali")

        # Tab Grafici: sotto-tab (chart + instant)
        tabPlots = QtWidgets.QTabWidget()

        self.tabChart = QtWidgets.QWidget()
        vchart = QtWidgets.QVBoxLayout(self.tabChart)
        self.pgChart = pg.PlotWidget(title="Chart concatenato (decimato)")
        self._chart_legend = pg.LegendItem()
        self._chart_legend.setParentItem(self.pgChart.getPlotItem().graphicsItem())
        self._chart_legend.anchor((0, 1), (0, 1), offset=(10, -10))
        vchart.addWidget(self.pgChart, 1)
        # Etichetta per mostrare il valore medio di ciascun canale attivo
        self.lblAvgChart = QtWidgets.QLabel("")
        # Riduce leggermente la dimensione del carattere per la stringa di media
        try:
            fnt_avg = self.lblAvgChart.font()
            fnt_avg.setPointSize(max(8, fnt_avg.pointSize() - 1))
            self.lblAvgChart.setFont(fnt_avg)
        except Exception:
            pass
        self.lblAvgChart.setWordWrap(True)
        vchart.addWidget(self.lblAvgChart)
        hctrl = QtWidgets.QHBoxLayout()
        self.btnClearChart = QtWidgets.QPushButton("Pulizia grafico")
        hctrl.addStretch(1)
        hctrl.addWidget(self.btnClearChart)
        vchart.addLayout(hctrl)
        tabPlots.addTab(self.tabChart, "Chart concatenato")

        self.tabInstant = QtWidgets.QWidget()
        vinst = QtWidgets.QVBoxLayout(self.tabInstant)
        self.pgInstant = pg.PlotWidget(title="Ultimo blocco (non concatenato)")
        self._instant_legend = pg.LegendItem()
        self._instant_legend.setParentItem(self.pgInstant.getPlotItem().graphicsItem())
        self._instant_legend.anchor((0, 1), (0, 1), offset=(10, -10))
        vinst.addWidget(self.pgInstant, 1)
        self.chkInstantView = QtWidgets.QCheckBox("Abilita visualizzazione")
        self.chkInstantView.setChecked(False)
        self.chkInstantView.toggled.connect(self._on_instant_view_toggled)
        vinst.addWidget(self.chkInstantView)
        tabPlots.addTab(self.tabInstant, "Blocchi istantanei")

        # --------------------------------------------------------------
        # FFT tab: mostra il modulo della trasformata di Fourier calcolata
        # sulla finestra temporale del chart concatenato decimato.
        # Ogni canale ? disegnato con colore e legenda dedicati.
        self.tabFFT = QtWidgets.QWidget()
        vfft = QtWidgets.QVBoxLayout(self.tabFFT)
        # Plot widget for the FFT magnitude spectra.  The title will be
        # updated dynamically based on the computation window.
        self.pgFFT = pg.PlotWidget(title="Spettro FFT")
        # Legend dedicated to the FFT plot.  Anchored in the upper left
        # corner with a slight offset to avoid overlapping the plot area.
        self._fft_legend = pg.LegendItem()
        try:
            self._fft_legend.setParentItem(self.pgFFT.getPlotItem().graphicsItem())
            self._fft_legend.anchor((0, 1), (0, 1), offset=(10, -10))
        except Exception:
            pass
        vfft.addWidget(self.pgFFT, 1)
        # Controls for FFT computation: duration and log-scale toggle
        ctrl_layout = QtWidgets.QHBoxLayout()
        # Enable/disable continuous FFT update from the decimated chart tail.
        self.chkFftEnable = QtWidgets.QCheckBox("Abilita FFT")
        self.chkFftEnable.toggled.connect(self._on_fft_enable_toggled)
        ctrl_layout.addWidget(self.chkFftEnable)
        ctrl_layout.addWidget(QtWidgets.QLabel("Durata FFT [s]:"))
        self.spinFftDuration = QtWidgets.QDoubleSpinBox()
        self.spinFftDuration.setDecimals(2)
        self.spinFftDuration.setMinimum(0.1)
        self.spinFftDuration.setMaximum(3600.0)
        self.spinFftDuration.setSingleStep(0.1)
        self.spinFftDuration.setValue(self._fft_duration_seconds)
        # When the user edits the duration, update internal state and
        # resize ring buffers accordingly.
        self.spinFftDuration.valueChanged.connect(self._on_fft_duration_changed)
        ctrl_layout.addWidget(self.spinFftDuration)
        self.chkFftLogScale = QtWidgets.QCheckBox("Scala log-log")
        # When toggled, adjust the plot axes to log or linear mode.
        self.chkFftLogScale.toggled.connect(self._on_fft_log_scale_changed)
        ctrl_layout.addWidget(self.chkFftLogScale)
        ctrl_layout.addStretch(1)
        self.lblFftStatus = QtWidgets.QLabel("")
        try:
            self.lblFftStatus.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            fnt_status = self.lblFftStatus.font()
            fnt_status.setPointSize(max(fnt_status.pointSize() - 1, 8))
            self.lblFftStatus.setFont(fnt_status)
        except Exception:
            pass
        ctrl_layout.addWidget(self.lblFftStatus)
        vfft.addLayout(ctrl_layout)
        # Manual axis limits for FFT plot.
        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.addWidget(QtWidgets.QLabel("X min:"))
        self.spinFftXMin = QtWidgets.QDoubleSpinBox()
        self.spinFftXMin.setRange(-1e9, 1e9)
        self.spinFftXMin.setDecimals(6)
        self.spinFftXMin.setValue(0.0)
        axis_layout.addWidget(self.spinFftXMin)
        axis_layout.addWidget(QtWidgets.QLabel("X max:"))
        self.spinFftXMax = QtWidgets.QDoubleSpinBox()
        self.spinFftXMax.setRange(-1e9, 1e9)
        self.spinFftXMax.setDecimals(6)
        self.spinFftXMax.setValue(50.0)
        axis_layout.addWidget(self.spinFftXMax)
        axis_layout.addWidget(QtWidgets.QLabel("Y min:"))
        self.spinFftYMin = QtWidgets.QDoubleSpinBox()
        self.spinFftYMin.setRange(-1e12, 1e12)
        self.spinFftYMin.setDecimals(6)
        self.spinFftYMin.setValue(0.0)
        axis_layout.addWidget(self.spinFftYMin)
        axis_layout.addWidget(QtWidgets.QLabel("Y max:"))
        self.spinFftYMax = QtWidgets.QDoubleSpinBox()
        self.spinFftYMax.setRange(-1e12, 1e12)
        self.spinFftYMax.setDecimals(6)
        self.spinFftYMax.setValue(10.0)
        axis_layout.addWidget(self.spinFftYMax)
        self.btnFftApplyAxes = QtWidgets.QPushButton("Applica assi FFT")
        self.btnFftAutoAxes = QtWidgets.QPushButton("Auto assi FFT")
        self.btnFftApplyAxes.clicked.connect(self._apply_fft_axis_limits)
        self.btnFftAutoAxes.clicked.connect(self._auto_fft_axis_limits)
        axis_layout.addWidget(self.btnFftApplyAxes)
        axis_layout.addWidget(self.btnFftAutoAxes)
        axis_layout.addStretch(1)
        vfft.addLayout(axis_layout)
        # Label to display peak information for each channel.  It will show
        # the amplitude and frequency of the dominant component of the
        # spectrum.  Word wrap ensures long text wraps across lines.
        self.lblFftPeakInfo = QtWidgets.QLabel("")
        self.lblFftPeakInfo.setWordWrap(True)
        font = self.lblFftPeakInfo.font()
        font.setPointSize(max(font.pointSize() - 1, 8))
        self.lblFftPeakInfo.setFont(font)
        vfft.addWidget(self.lblFftPeakInfo)
        tabPlots.addTab(self.tabFFT, "FFT")

        self.tabs.addTab(tabPlots, "Grafici")

        # Barra salvataggio in basso
        bottom = QtWidgets.QHBoxLayout()
        self.txtSaveDir = QtWidgets.QLineEdit(self._save_dir)
        self.btnBrowseDir = QtWidgets.QPushButton("Sfoglia cartella...")
        self.txtBaseName = QtWidgets.QLineEdit(self._base_filename)
        # SpinBox per impostare la dimensione del buffer in RAM (MB) per il salvataggio
        self.spinRam = QtWidgets.QSpinBox()
        # Limiti ragionevoli: da 10 MB fino a 16 GB
        self.spinRam.setRange(10, 16384)
        # Valore di default basato sul limite corrente dell'acquisition manager
        try:
            _u, _lim = self.acq.get_memory_usage()
            self.spinRam.setValue(max(1, int(_lim / (1024 * 1024))))
        except Exception:
            self.spinRam.setValue(500)
        self.spinRam.setSuffix(" MB")
        self.spinRam.setSingleStep(50)
        self.btnStart = QtWidgets.QPushButton("Salva dati")            # passa a "Salvo in (xx s)?â€
        self.btnStop = QtWidgets.QPushButton("Stop e ricomponi...")
        self.btnStop.setEnabled(False)

        bottom.addWidget(QtWidgets.QLabel("Percorso salvataggio:"))
        bottom.addWidget(self.txtSaveDir, 1)
        bottom.addWidget(self.btnBrowseDir)
        bottom.addSpacing(12)
        bottom.addWidget(QtWidgets.QLabel("Nome file:"))
        bottom.addWidget(self.txtBaseName)
        # Controllo per la dimensione del buffer in RAM
        bottom.addSpacing(12)
        bottom.addWidget(QtWidgets.QLabel("Buffer RAM:"))
        bottom.addWidget(self.spinRam)
        bottom.addStretch(1)
        bottom.addWidget(self.btnStart)
        bottom.addWidget(self.btnStop)
        main.addLayout(bottom)

        # Timer per l'aggiornamento dei grafici.  Un intervallo pi? lungo
        # (100 ms invece dei 50 ms precedenti) riduce il numero di
        # conversioni da deque a array e di chiamate a setData, riducendo
        # l'uso di memoria nel lungo periodo.  Questo valore pu? essere
        # ulteriormente modificato dinamicamente dalla routine di controllo
        # dello stall.
        self.guiTimer = QtCore.QTimer(self)
        self.guiTimer.setInterval(100)
        self.guiTimer.timeout.connect(self._refresh_plots)

        # Status bar + etichetta sempre visibile con rate
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.lblRateInfo = QtWidgets.QLabel("-")
        our_font = self.lblRateInfo.font()
        our_font.setPointSize(9)
        self.lblRateInfo.setFont(our_font)
        self.statusBar.addPermanentWidget(self.lblRateInfo)

    # ------------------------- Configuration persistence -------------------------
    def _load_config(self):
        """
        Load persistent settings from a JSON file if it exists. The settings
        include the last used save directory, base filename, memory buffer
        size and sampling rate. This method should be called after the UI
        widgets have been created so that values can be applied.
        """
        if not os.path.isfile(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return
        # Apply known settings if present
        try:
            sd = cfg.get("save_dir")
            if sd:
                self._save_dir = sd
                self.txtSaveDir.setText(sd)
        except Exception:
            pass
        try:
            bn = cfg.get("base_filename")
            if bn:
                self._base_filename = bn
                self.txtBaseName.setText(bn)
        except Exception:
            pass
        try:
            ram_mb = int(cfg.get("ram_mb", 0))
            if ram_mb > 0:
                self.spinRam.setValue(ram_mb)
        except Exception:
            pass
        try:
            fs = cfg.get("fs")
            if fs:
                # Show the saved sampling rate in the rateEdit field
                self.rateEdit.setText(str(fs))
        except Exception:
            pass

    def _save_config(self):
        """
        Save current UI settings to a JSON configuration file. Only basic
        values (save directory, base filename, buffer size and sample rate)
        are persisted. This method is called automatically on application
        close.
        """
        cfg = {}
        try:
            cfg["save_dir"] = self.txtSaveDir.text().strip()
        except Exception:
            pass
        try:
            cfg["base_filename"] = self.txtBaseName.text().strip()
        except Exception:
            pass
        try:
            cfg["ram_mb"] = int(self.spinRam.value())
        except Exception:
            pass
        try:
            txt = self.rateEdit.text().strip()
            if txt:
                cfg["fs"] = float(txt)
        except Exception:
            pass
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    # -------------------------- Backlog/Disk stall check --------------------------
    def _check_backlog(self):
        """
        Periodically monitor the estimated disk write backlog. If the backlog
        exceeds a predefined threshold, a warning is shown in the status bar
        and the chart refresh interval is reduced to minimize CPU overhead.
        When the backlog drops below the threshold, the refresh interval is
        restored and the warning message cleared.
        """
        try:
            # Only monitor when recording is active
            if not self.acq.recording_enabled:
                # When not recording, ensure GUI timer uses default interval and clear warning
                if self._stall_active:
                    self.guiTimer.setInterval(self._default_gui_interval)
                    self.statusBar.showMessage("")
                    self._stall_active = False
                return
            backlog_mb = 0.0
            try:
                backlog_mb = float(self.acq.get_backlog_estimate())
            except Exception:
                backlog_mb = 0.0
            # Threshold for disk stall warning (MB)
            threshold_mb = 200.0
            if backlog_mb >= threshold_mb and not self._stall_active:
                # Enter stall mode: slow down UI updates and display warning
                self._stall_active = True
                # Reduce chart refresh frequency to ease CPU and I/O pressure
                try:
                    self.guiTimer.setInterval(max(self._default_gui_interval, 200))
                except Exception:
                    pass
                msg = f"DISK STALL: backlog {backlog_mb:.0f} MB. Rallento l'aggiornamento grafici per evitare perdite."
                self.statusBar.showMessage(msg)
            elif backlog_mb < threshold_mb and self._stall_active:
                # Exit stall mode
                self._stall_active = False
                try:
                    self.guiTimer.setInterval(self._default_gui_interval)
                except Exception:
                    pass
                self.statusBar.showMessage("")
        except Exception:
            pass

    def _connect_signals(self):
        # pulsanti
        self.btnRefresh.clicked.connect(self.refresh_devices)
        self.btnBrowseDir.clicked.connect(self._choose_folder)
        self.btnStart.clicked.connect(self._on_start_saving)
        self.btnStop.clicked.connect(self._on_stop)
        self.btnClearChart.clicked.connect(self._clear_chart)
        self.btnDefineTypes.clicked.connect(self._open_resource_manager)

        # collegamenti per salvataggio/caricamento workspace
        try:
            self.btnSaveWorkspace.clicked.connect(self._save_workspace)
            self.btnLoadWorkspace.clicked.connect(self._load_workspace)
        except Exception:
            pass

        # tabella: prima puliamo eventuali collegamenti generici che
        # potrebbero riconfigurare anche quando si rinomina
        try:
            self.table.cellChanged.disconnect()
        except Exception:
            pass
        try:
            self.table.itemChanged.disconnect()
        except Exception:
            pass
        self.table.itemChanged.connect(self._on_table_item_changed)

        self.cmbDevice.currentIndexChanged.connect(self._on_device_changed)

        # Aggiorna la frequenza di campionamento quando l'utente conferma il valore
        try:
            self.rateEdit.editingFinished.connect(self._on_rate_edit_finished)
        except Exception:
            pass

        # callback dal core (rimappati in segnali Qt)
        self.channelValueUpdated.connect(self._update_table_value)
        self.sigInstantBlock.connect(self._slot_instant_block)
        self.sigChartPoints.connect(self._slot_chart_points)

        self.acq.on_channel_value = lambda name, val: self.channelValueUpdated.emit(name, val)
        self.acq.on_new_instant_block = lambda t, ys, names: self.sigInstantBlock.emit(t, ys, names)
        self.acq.on_new_chart_points = lambda t_pts, ys_pts, names: self.sigChartPoints.emit(t_pts, ys_pts, names)

    def _init_sync_agent(self):
        if ModuleSyncAgent is None:
            return
        try:
            agent = ModuleSyncAgent(self)
        except Exception:
            return
        if not agent.is_active():
            return

        self._sync_agent = agent
        self._sync_agent.register_handler("APPLY_CHASSIS_CONFIG", self._sync_cmd_apply_chassis_config)
        self._sync_agent.register_handler("STATUS_SNAPSHOT", self._sync_cmd_status_snapshot)
        self._sync_agent.register_handler("PREPARE_SAVE", self._sync_cmd_prepare_save)
        self._sync_agent.register_handler("CONFIGURE_SYNC", self._sync_cmd_configure_sync)
        self._sync_agent.register_handler("ARM_ACQUISITION", self._sync_cmd_arm_acquisition)
        self._sync_agent.register_handler("START_SYNC", self._sync_cmd_start_sync)
        self._sync_agent.register_handler("START_SAVE", self._sync_cmd_start_save)
        self._sync_agent.register_handler("SET_SYNC_WRITE_CUTOFF", self._sync_cmd_set_sync_write_cutoff)
        self._sync_agent.register_handler("STOP_AND_MERGE", self._sync_cmd_stop_and_merge)
        self._sync_agent.register_handler("UNLOCK_UI", self._sync_cmd_unlock_ui)
        self._sync_agent.register_handler("ABORT_PREPARED", self._sync_cmd_abort_prepared)
        self._sync_agent.register_handler("SHUTDOWN", self._sync_cmd_shutdown)
        self._sync_agent.start(
            {
                "board": "NI9234",
                "pid": int(os.getpid()),
                "device_name": str(self._forced_device_name_from_env() or ""),
            }
        )
        QtCore.QTimer.singleShot(0, self._sync_emit_status_update)

    def _sync_status_snapshot(self) -> Dict[str, Any]:
        device_name = str((self.cmbDevice.currentData() or self.cmbDevice.currentText() or "").strip())
        phys, _labels = self._enabled_phys_and_labels()
        active_channels = int(len(phys))
        fs_max_hz = 0.0
        if device_name and active_channels > 0:
            try:
                fs_max_hz = float(self.acq._compute_per_channel_rate(device_name, active_channels))
            except Exception:
                try:
                    fs_max_hz = float(self.acq.current_rate_hz or 0.0)
                except Exception:
                    fs_max_hz = 0.0
        return {
            "module_id": str(self._sync_agent.module_id() if self._sync_agent is not None else ""),
            "device_name": device_name,
            "active_channels": active_channels,
            "fs_max_hz": float(fs_max_hz if fs_max_hz > 0 else 0.0),
            "current_rate_hz": float(getattr(self.acq, "current_rate_hz", 0.0) or 0.0),
            "is_simulated": self._is_current_device_simulated(),
            "running": bool(getattr(self.acq, "_running", False)),
            "recording": bool(getattr(self.acq, "recording_enabled", False)),
            "samples_acquired": int(getattr(self.acq, "_global_samples", 0) or 0),
        }

    def _sync_emit_status_update(self):
        if self._sync_agent is None:
            return
        try:
            self._sync_agent.send_event("STATUS_UPDATE", self._sync_status_snapshot())
        except Exception:
            pass

    def _sync_cmd_status_snapshot(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._sync_status_snapshot()

    def _is_current_device_simulated(self) -> bool:
        txt = str(self.cmbDevice.currentText() or "")
        return "[SIMULATED]" in txt.upper()

    def _set_remote_control_lock(self, lock: bool):
        lock = bool(lock)
        self._set_table_lock(lock)
        self.txtSaveDir.setEnabled(not lock)
        self.btnBrowseDir.setEnabled(not lock)
        self.txtBaseName.setEnabled(not lock)
        self.spinRam.setEnabled(not lock)
        self.rateEdit.setEnabled(not lock)
        self.btnDefineTypes.setEnabled(not lock)
        self.btnSaveWorkspace.setEnabled(not lock)
        self.btnLoadWorkspace.setEnabled(not lock)
        self.chkFftEnable.setEnabled(not lock)
        self.spinFftDuration.setEnabled(not lock)
        forced_device_name = self._forced_device_name_from_env()
        self.cmbDevice.setEnabled((not lock) and (not bool(forced_device_name)))
        self.btnRefresh.setEnabled((not lock) and (not bool(forced_device_name)))
        if lock:
            self.btnStart.setEnabled(False)
            self.btnStop.setEnabled(False)
        else:
            self.btnStart.setEnabled(not bool(self.acq.recording_enabled))
            self.btnStop.setEnabled(bool(self.acq.recording_enabled))

    def _run_without_message_boxes(self, fn):
        orig_info = QtWidgets.QMessageBox.information
        orig_warn = QtWidgets.QMessageBox.warning
        orig_crit = QtWidgets.QMessageBox.critical

        def _silent(*_args, **_kwargs):
            return QtWidgets.QMessageBox.Ok

        try:
            QtWidgets.QMessageBox.information = _silent
            QtWidgets.QMessageBox.warning = _silent
            QtWidgets.QMessageBox.critical = _silent
            return fn()
        finally:
            QtWidgets.QMessageBox.information = orig_info
            QtWidgets.QMessageBox.warning = orig_warn
            QtWidgets.QMessageBox.critical = orig_crit

    def _find_latest_tdms(self) -> str:
        base_dir = str(self._save_dir or "").strip()
        if not base_dir or not os.path.isdir(base_dir):
            return ""
        files = glob.glob(os.path.join(base_dir, "*.tdms"))
        if not files:
            return ""
        try:
            files.sort(key=lambda p: os.path.getmtime(p))
        except Exception:
            files.sort()
        return files[-1]

    def _sync_cmd_apply_chassis_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        save_dir = str(payload.get("save_dir", "") or "").strip()
        base_name = str(payload.get("base_filename", "") or "").strip()
        fs_hz = float(payload.get("fs_hz", 0.0) or 0.0)
        ram_mb = payload.get("ram_mb", None)

        if save_dir:
            self.txtSaveDir.setText(save_dir)
        if base_name:
            if not base_name.lower().endswith(".tdms"):
                base_name += ".tdms"
            self.txtBaseName.setText(base_name)
        if ram_mb is not None:
            try:
                ram_i = int(float(ram_mb))
                ram_i = max(self.spinRam.minimum(), min(self.spinRam.maximum(), ram_i))
                self.spinRam.setValue(ram_i)
            except Exception:
                pass
        if fs_hz > 0:
            self.rateEdit.setText(str(int(round(fs_hz))))
            self._run_without_message_boxes(self._on_rate_edit_finished)

        snap = self._sync_status_snapshot()
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_prepare_save(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._device_ready:
            raise RuntimeError("Dispositivo non pronto.")
        phys, _labels = self._enabled_phys_and_labels()
        if not phys:
            snap = self._sync_status_snapshot()
            snap["eligible"] = False
            self._sync_emit_status_update()
            return snap
        if not bool(getattr(self.acq, "_running", False)):
            self._run_without_message_boxes(self._update_acquisition_from_table)
        if not bool(getattr(self.acq, "_running", False)):
            raise RuntimeError("Acquisizione non avviata.")
        snap = self._sync_status_snapshot()
        snap["eligible"] = True
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_configure_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fs_hz = float(payload.get("fs_hz", 0.0) or 0.0)
        if fs_hz > 0:
            self.rateEdit.setText(str(int(round(fs_hz))))
            self._run_without_message_boxes(self._on_rate_edit_finished)
        snap = self._sync_status_snapshot()
        snap["hardware_supported"] = bool(not self._is_current_device_simulated())
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_arm_acquisition(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fs_hz = float(payload.get("fs_hz", 0.0) or 0.0)
        if fs_hz > 0:
            self.rateEdit.setText(str(int(round(fs_hz))))
            self._run_without_message_boxes(self._on_rate_edit_finished)
        try:
            if bool(getattr(self.acq, "_running", False)):
                self.acq.stop()
        except Exception:
            pass
        phys, _labels = self._enabled_phys_and_labels()
        if not phys:
            raise RuntimeError("Nessun canale abilitato per arm.")
        self._sync_arm_requested = True
        snap = self._sync_status_snapshot()
        snap["armed"] = True
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_start_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._sync_arm_requested:
            raise RuntimeError("Modulo non armato.")
        self._set_table_lock(True)
        start_save_on_sync = bool(payload.get("start_save_on_sync", False))
        sync_anchor_epoch_ns = int(payload.get("sync_anchor_epoch_ns", 0) or 0)
        if sync_anchor_epoch_ns > 0:
            try:
                self.acq.set_sync_anchor_epoch_s(float(sync_anchor_epoch_ns) / 1e9)
            except Exception:
                pass
        mode = str(payload.get("mode", "software") or "software").strip().lower()
        sync_role = str(payload.get("sync_role", "") or "").strip().lower()
        trig_src = str(payload.get("start_trigger_source", "") or "").strip()
        if mode == "hardware":
            self._pending_sync_start_cfg = {
                "sync_mode": "hardware",
                "sync_role": sync_role,
                "start_trigger_source": trig_src,
            }
        else:
            self._pending_sync_start_cfg = None
        start_epoch_ns = int(payload.get("start_epoch_ns", 0) or 0)
        if start_epoch_ns > 0 and (mode != "hardware" or sync_role == "master"):
            while True:
                now = time.time_ns()
                dt_ns = start_epoch_ns - now
                if dt_ns <= 0:
                    break
                if dt_ns > 3_000_000:
                    time.sleep((dt_ns - 1_000_000) / 1_000_000_000.0)
                else:
                    break
            while time.time_ns() < start_epoch_ns:
                pass
        try:
            self._run_without_message_boxes(self._update_acquisition_from_table)
        finally:
            self._pending_sync_start_cfg = None
        if not bool(getattr(self.acq, "_running", False)):
            raise RuntimeError("Start sincronizzato fallito.")
        if start_save_on_sync:
            self._sync_remote_active = True
            self._set_remote_control_lock(True)
            # Hold writer until root computes common N0 and commits cutoff.
            try:
                self.acq.set_sync_write_start_index(10**12)
            except Exception:
                pass
            self._run_without_message_boxes(self._on_start_saving)
            if not bool(self.acq.recording_enabled):
                self._sync_remote_active = False
                self._set_remote_control_lock(False)
                raise RuntimeError("Salvataggio sync non avviato.")
        self._sync_arm_requested = False
        snap = self._sync_status_snapshot()
        snap["running"] = True
        snap["start_ns"] = time.time_ns()
        if start_save_on_sync:
            snap["recording"] = bool(self.acq.recording_enabled)
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_set_sync_write_cutoff(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cutoff = int(payload.get("cutoff_index", 0) or 0)
        except Exception:
            cutoff = 0
        try:
            self.acq.set_sync_write_start_index(cutoff)
        except Exception:
            pass
        snap = self._sync_status_snapshot()
        snap["sync_cutoff_index"] = int(cutoff)
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_start_save(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        start_epoch_ns = int(_payload.get("start_epoch_ns", 0) or 0)
        if start_epoch_ns > 0:
            while True:
                now = time.time_ns()
                dt_ns = start_epoch_ns - now
                if dt_ns <= 0:
                    break
                if dt_ns > 3_000_000:
                    time.sleep((dt_ns - 1_000_000) / 1_000_000_000.0)
                else:
                    break
            while time.time_ns() < start_epoch_ns:
                pass
        self._sync_remote_active = True
        self._set_remote_control_lock(True)
        self._run_without_message_boxes(self._on_start_saving)
        if not bool(self.acq.recording_enabled):
            self._sync_remote_active = False
            self._set_remote_control_lock(False)
            raise RuntimeError("Salvataggio non avviato.")
        snap = self._sync_status_snapshot()
        snap["recording"] = True
        if start_epoch_ns > 0:
            snap["requested_start_ns"] = int(start_epoch_ns)
        snap["recording_start_ns"] = int(time.time_ns())
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_stop_and_merge(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        before = self._find_latest_tdms()
        self._run_without_message_boxes(self._on_stop)
        after = self._find_latest_tdms()
        self._sync_remote_active = False
        self._sync_arm_requested = False
        snap = self._sync_status_snapshot()
        snap["recording"] = bool(self.acq.recording_enabled)
        snap["final_tdms"] = after or before
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_unlock_ui(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        self._sync_remote_active = False
        self._sync_arm_requested = False
        self._set_remote_control_lock(False)
        snap = self._sync_status_snapshot()
        snap["unlocked"] = True
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_abort_prepared(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.acq.set_recording(False)
        except Exception:
            pass
        try:
            self.acq.set_sync_write_start_index(0)
            self.acq.set_sync_anchor_epoch_s(None)
        except Exception:
            pass
        try:
            self.acq.stop()
        except Exception:
            pass
        self._sync_remote_active = False
        self._sync_arm_requested = False
        self._set_remote_control_lock(False)
        snap = self._sync_status_snapshot()
        snap["aborted"] = True
        self._sync_emit_status_update()
        return snap

    def _sync_cmd_shutdown(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            QtCore.QMetaObject.invokeMethod(self, "close", QtCore.Qt.QueuedConnection)
        except Exception:
            QtCore.QTimer.singleShot(0, self.close)
        return {"shutdown": True}

    # ----------------------------- Devices -----------------------------
    def refresh_devices(self):
        """
        Popola la combo dispositivi includendo anche i moduli simulati.
        Se sono presenti piu moduli, apre un dialog per scegliere.
        """
        try:
            metas = self.acq.list_current_devices_meta()
        except AttributeError:
            try:
                names = self.acq.list_ni9234_devices()
            except Exception:
                names = []
            metas = [{"name": n, "product_type": "NI 9234", "is_simulated": False,
                      "chassis": n.split("Mod")[0] if "Mod" in n else ""} for n in names]
        except Exception:
            metas = []

        metas, preferred_idx = self._prioritize_preferred_devices(metas)
        forced_device_name = self._forced_device_name_from_env()
        if forced_device_name:
            forced_idx = -1
            for idx, meta in enumerate(metas):
                if str(meta.get("name", "") or "").strip() == forced_device_name:
                    forced_idx = idx
                    break

            if forced_idx < 0:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Dispositivo non disponibile",
                    f'Il dispositivo assegnato "{forced_device_name}" non e disponibile.',
                )
                self._abort_startup_on_device_cancel()
                return

            if forced_idx > 0:
                ordered = list(metas)
                target = ordered.pop(forced_idx)
                ordered.insert(0, target)
                metas = ordered
                preferred_idx = 0

        self.cmbDevice.blockSignals(True)
        self.cmbDevice.clear()
        for m in metas:
            name = m.get("name", "?")
            pt = m.get("product_type", "")
            ch = m.get("chassis", "")
            sim = " [SIMULATED]" if m.get("is_simulated") else ""
            label = f"{name} - {pt} - ({ch}){sim}" if ch else f"{name} - {pt}{sim}"
            self.cmbDevice.addItem(label, userData=name)
        self.cmbDevice.blockSignals(False)

        self._device_ready = bool(metas)

        if not metas:
            QtWidgets.QMessageBox.information(
                self, "Nessun dispositivo",
                "Nessun NI-9234 trovato. Verifica in NI-MAX (anche simulati)."
            )
            if forced_device_name:
                self._abort_startup_on_device_cancel()
                return
        elif forced_device_name:
            self.cmbDevice.setCurrentIndex(0)
        elif len(metas) == 1:
            self.cmbDevice.setCurrentIndex(0)
        else:
            chosen = self._prompt_device_choice(metas, preferred_idx=preferred_idx)
            if chosen:
                for i in range(self.cmbDevice.count()):
                    if self.cmbDevice.itemData(i) == chosen:
                        self.cmbDevice.setCurrentIndex(i)
                        break
            else:
                self._abort_startup_on_device_cancel()
                return

        # Se il processo e stato avviato per un device specifico, vincola la selezione.
        self.cmbDevice.setEnabled(not bool(forced_device_name))
        self.btnRefresh.setEnabled(not bool(forced_device_name))

        self._populate_table()
        self._populate_type_column()
        self._recompute_all_calibrations()
        self.lblRateInfo.setText("-")
        self._sync_emit_status_update()

    def _abort_startup_on_device_cancel(self):
        # Chiusura pulita del modulo: evita sys.exit immediato mentre Qt
        # ha thread/eventi ancora attivi (es. worker FFT).
        self._device_ready = False
        QtCore.QTimer.singleShot(0, self.close)

    def _preferred_chassis_from_env(self):
        alias = str(os.environ.get("CDAQ_SELECTED_ALIAS", "") or "").strip()
        raw = str(os.environ.get("CDAQ_SELECTED_SIMULATED", "") or "").strip().lower()

        is_sim = None
        if raw in ("1", "true", "yes"):
            is_sim = True
        elif raw in ("0", "false", "no"):
            is_sim = False

        return alias, is_sim

    def _forced_device_name_from_env(self):
        return str(os.environ.get("CDAQ_TARGET_DEVICE_NAME", "") or "").strip()

    def _prioritize_preferred_devices(self, metas):
        if not metas:
            return metas, 0

        preferred_alias, preferred_is_sim = self._preferred_chassis_from_env()
        if not preferred_alias:
            return metas, 0

        preferred_idx = -1
        target_alias = preferred_alias.lower()

        for idx, meta in enumerate(metas):
            chassis_alias = str(meta.get("chassis", "") or "").strip().lower()
            if chassis_alias != target_alias:
                continue
            is_sim = bool(meta.get("is_simulated"))
            if preferred_is_sim is None or is_sim == preferred_is_sim:
                preferred_idx = idx
                break

        if preferred_idx <= 0:
            return metas, 0

        ordered = list(metas)
        preferred_meta = ordered.pop(preferred_idx)
        ordered.insert(0, preferred_meta)
        return ordered, 0

    def _prompt_device_choice(self, metas, preferred_idx=0):
        items = []
        for m in metas:
            name = m.get("name", "?")
            pt = m.get("product_type", "")
            ch = m.get("chassis", "")
            sim = " [SIMULATED]" if m.get("is_simulated") else ""
            label = f"{name} - {pt} - ({ch}){sim}" if ch else f"{name} - {pt}{sim}"
            items.append(label)

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Seleziona dispositivo")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Sono presenti piu moduli NI-9234.\nScegli quello da usare:"))

        cmb = QtWidgets.QComboBox(dialog)
        cmb.addItems(items)
        if cmb.count() > 0:
            preferred_idx = max(0, min(int(preferred_idx or 0), cmb.count() - 1))
            cmb.setCurrentIndex(preferred_idx)
        else:
            preferred_idx = 0
        layout.addWidget(cmb)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None

        chosen_idx = cmb.currentIndex()
        if chosen_idx < 0 or chosen_idx >= len(metas):
            return None
        return str(metas[chosen_idx].get("name", "") or "")

    def _on_device_changed(self, _):
        self._stop_acquisition_ui_only()
        self._reset_plots()
        self.lblRateInfo.setText("-")

    # ----------------------------- Sensor defs -----------------------------
    def _read_sensor_defs(self) -> dict:
        """
        Legge il file XML (multi-punti o 2-punti) e ritorna dict:
            name -> {unit, a, b}
        """
        defs = {}
        # Use the current sensor DB path if available; fallback to default
        try:
            p = self._sensor_db_path
        except Exception:
            p = SENSOR_DB_DEFAULT
        if not os.path.isfile(p):
            return defs
        # Determine supported DAQ models for this application. The program
        # must load only sensors compatible with the current board type.
        try:
            bt = getattr(self.acq, "board_type", "NI9234")
        except Exception:
            bt = "NI9234"
        board_tag = str(bt or "NI9234").strip().upper()
        try:
            tree = ET.parse(p)
            root = tree.getroot()
            for s in root.findall(XML_ITEM):
                try:
                    name = (s.findtext(XML_NAME, default="") or "").strip()
                except Exception:
                    continue
                if not name:
                    continue
                # Read unit and supportedDAQ (if present)
                try:
                    unit = (s.findtext(XML_UNIT, default="") or "").strip()
                except Exception:
                    unit = ""
                # Check supportedDAQ
                try:
                    supported = (s.findtext("supportedDAQ", default="") or "").strip()
                    # If a supportedDAQ tag is present, ensure the current board appears among the comma-separated values
                    if supported:
                        items = [x.strip().upper() for x in supported.split(",") if x.strip()]
                        if all(item != board_tag for item in items):
                            continue
                    else:
                        # If the tag does not exist or is empty, do not include the sensor
                        continue
                except Exception:
                    # In caso di errore, non includere il sensore
                    continue
                # nuovo schema multi-punti
                cal = s.find(XML_CAL)
                if cal is not None:
                    V, X = [], []
                    for pt in cal.findall(XML_POINT):
                        try:
                            v = float(pt.get("volt", "nan"))
                            x = float(pt.get("value", "nan"))
                        except Exception:
                            continue
                        if np.isfinite(v) and np.isfinite(x):
                            V.append(v); X.append(x)
                    if len(V) >= 2:
                        A = np.vstack([np.asarray(V), np.ones(len(V))]).T
                        a, b = np.linalg.lstsq(A, np.asarray(X), rcond=None)[0]
                        defs[name] = {"unit": unit, "a": float(a), "b": float(b)}
                        continue
                # compat vecchio schema (2 punti)
                def _f(tag):
                    try: return float(s.findtext(tag, default="0") or "0")
                    except Exception: return 0.0
                v1v = _f(XML_V1V); x1 = _f(XML_V1)
                v2v = _f(XML_V2V); x2 = _f(XML_V2)
                if abs(v2v - v1v) > 1e-12:
                    a = (x2 - x1) / (v2v - v1v)
                    b = x1 - a * v1v
                else:
                    a, b = 1.0, 0.0
                defs[name] = {"unit": unit, "a": a, "b": b}
        except Exception:
            pass
        return defs

    def _populate_type_column(self):
        """Popola 'Tipo risorsa' con: Voltage + nomi da Sensor DB."""
        # Read sensor definitions filtered by supported DAQ and use keys as names
        names = []
        try:
            defs = self._read_sensor_defs()
            names = sorted(defs.keys())
        except Exception:
            names = []
        for r in range(self.table.rowCount()):
            cmb: QtWidgets.QComboBox = self.table.cellWidget(r, COL_TYPE)
            if not isinstance(cmb, QtWidgets.QComboBox):
                continue
            cur = cmb.currentText()
            cmb.blockSignals(True)
            cmb.clear()
            cmb.setEditable(False)
            cmb.addItem("Voltage")
            for n in names:
                cmb.addItem(n)
            if cur and cur != "Voltage":
                idx = cmb.findText(cur)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)
                else:
                    cmb.setCurrentIndex(0)
            else:
                cmb.setCurrentIndex(0)
            cmb.blockSignals(False)

    def _recompute_all_calibrations(self):
        defs = self._read_sensor_defs()
        for r in range(self.table.rowCount()):
            phys = self.table.item(r, COL_PHYS).text().strip()
            cmb: QtWidgets.QComboBox = self.table.cellWidget(r, COL_TYPE)
            chosen = cmb.currentText().strip() if cmb else "Voltage"
            if chosen == "Voltage" or chosen == "":
                self._calib_by_phys[phys] = {"unit":"", "a":1.0, "b":0.0}
            else:
                self._calib_by_phys[phys] = defs.get(chosen, {"unit":"", "a":1.0, "b":0.0})
        self._rebuild_legends()
        self._push_sensor_map_to_core()

        # Dopo aver ricalcolato le calibrazioni e aggiornato le legende,
        # aggiorna anche i suffissi dei limiti per tutte le righe.  Questo
        # assicura che i campi Limite Max/Min visualizzino l'unit? corretta.
        try:
            self._update_limit_units_all()
        except Exception:
            pass

    def _push_sensor_map_to_core(self):
        mapping = {}
        for r in range(self.table.rowCount()):
            phys = self.table.item(r, COL_PHYS).text().strip()
            base_label = self.table.item(r, COL_LABEL).text().strip() or phys
            cal = self._calib_by_phys.get(phys, {"unit":"", "a":1.0, "b":0.0})
            unit = cal.get("unit",""); a = float(cal.get("a",1.0)); b = float(cal.get("b",0.0))
            display_label = f"{base_label} [{unit}]" if unit else base_label
            mapping[phys] = {
                "unit": unit, "a": a, "b": b,
                "sensor_name": self._sensor_type_by_phys.get(phys, "Voltage"),
                "display_label": display_label
            }
        try:
            self.acq.set_sensor_map(mapping)
        except Exception:
            pass

    # ---------------------- Aggiornamento suffissi limiti ----------------------
    def _update_limit_units_for_row(self, row: int):
        """
        Aggiorna il suffisso dei campi "Limite Max input" e "Limite Min input"
        per la riga specificata.  Utilizza l'unit? ingegneristica associata al
        sensore selezionato per quel canale.  Se non c'? unit? (ad esempio per
        "Voltage"), il suffisso viene rimosso.
        """
        # Recupera il nome fisico del canale
        try:
            item_phys = self.table.item(row, COL_PHYS)
            phys = item_phys.text().strip() if item_phys else ""
        except Exception:
            phys = ""
        if not phys:
            return
        # Determina l'unit? corrente
        unit = ""
        try:
            unit = self._calib_by_phys.get(phys, {}).get("unit", "")
        except Exception:
            unit = ""
        suffix = f" {unit}" if unit else ""
        # Aggiorna entrambi gli spinbox
        try:
            wmax = self.table.cellWidget(row, COL_LIMIT_MAX)
            if isinstance(wmax, QtWidgets.QDoubleSpinBox):
                wmax.setSuffix(suffix)
        except Exception:
            pass
        try:
            wmin = self.table.cellWidget(row, COL_LIMIT_MIN)
            if isinstance(wmin, QtWidgets.QDoubleSpinBox):
                wmin.setSuffix(suffix)
        except Exception:
            pass

    def _update_limit_units_all(self):
        """
        Aggiorna i suffissi dei campi Limite Max/Min per tutte le righe
        della tabella in base alle unit? ingegneristiche attualmente
        selezionate.  Ãˆ utile chiamare questo metodo dopo che sono state
        ricalcolate le calibrazioni o dopo la costruzione della tabella.
        """
        for r in range(self.table.rowCount()):
            try:
                self._update_limit_units_for_row(r)
            except Exception:
                pass

    # ----------------------------- Tabella -----------------------------
    def _populate_table(self):
        """
        Popola la tabella dei canali in base al numero di canali disponibili
        per la scheda corrente. Ogni riga rappresenta un canale fisico e
        contiene colonne per l'abilitazione, la selezione del tipo di
        risorsa (sensore), il nome del canale, il valore istantaneo, il
        reset dello zero, il valore azzerato, il coupling e i limiti
        dell'ingresso fisico.
        """
        self._building_table = True
        try:
            n = int(getattr(self.acq, "num_channels", 8))
        except Exception:
            n = 8
        self.table.setRowCount(n)
        for i in range(n):
            phys = f"ai{i}"
            # Abilita
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)
            self.table.setItem(i, COL_ENABLE, it)

            # Canale fisico (non modificabile)
            physItem = QtWidgets.QTableWidgetItem(phys)
            physItem.setFlags(physItem.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(i, COL_PHYS, physItem)

            # Tipo risorsa (sensore): selezione vincolata ai sensori supportati + Voltage
            cmbType = QtWidgets.QComboBox()
            cmbType.setEditable(False)
            cmbType.addItem("Voltage")
            cmbType.currentTextChanged.connect(lambda _t, row=i: self._type_changed_for_row(row))
            self.table.setCellWidget(i, COL_TYPE, cmbType)

            # Nome canale (label utente)
            labelItem = QtWidgets.QTableWidgetItem(self._label_by_phys.get(phys, phys))
            self.table.setItem(i, COL_LABEL, labelItem)

            # Valore istantaneo (solo display)
            valItem = QtWidgets.QTableWidgetItem("-")
            valItem.setFlags(valItem.flags() & ~QtCore.Qt.ItemIsUserCheckable & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(i, COL_VALUE, valItem)

            # Azzeramento: pulsante
            btnZero = QtWidgets.QPushButton("Azzeramento")
            btnZero.clicked.connect(lambda _, c=phys: self._on_zero_button_clicked(c))
            self.table.setCellWidget(i, COL_ZERO_BTN, btnZero)

            # Valore azzerato (display/placeholder)
            zeroItem = QtWidgets.QTableWidgetItem("0.0")
            zeroItem.setFlags(zeroItem.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(i, COL_ZERO_VAL, zeroItem)

            # Coupling: combobox con DC, AC, IEPE + AC
            cmbCoupling = QtWidgets.QComboBox()
            cmbCoupling.addItems(["DC", "AC", "IEPE + AC"])
            cmbCoupling.setCurrentIndex(0)
            cmbCoupling.currentTextChanged.connect(lambda _t, row=i: self._on_config_changed_for_row(row))
            self.table.setCellWidget(i, COL_COUPLING, cmbCoupling)

            # Limite Max input: spinbox
            spinMax = QtWidgets.QDoubleSpinBox()
            spinMax.setDecimals(9)
            spinMax.setRange(-1e12, 1e12)
            # Usa +5 come valore predefinito per il limite massimo
            spinMax.setValue(5.0)
            spinMax.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            spinMax.valueChanged.connect(lambda _v, row=i: self._on_config_changed_for_row(row))
            self.table.setCellWidget(i, COL_LIMIT_MAX, spinMax)

            # Limite Min input: spinbox
            spinMin = QtWidgets.QDoubleSpinBox()
            spinMin.setDecimals(9)
            spinMin.setRange(-1e12, 1e12)
            # Usa -5 come valore predefinito per il limite minimo
            spinMin.setValue(-5.0)
            spinMin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            spinMin.valueChanged.connect(lambda _v, row=i: self._on_config_changed_for_row(row))
            self.table.setCellWidget(i, COL_LIMIT_MIN, spinMin)

            # Inizializza la configurazione del canale nel core
            try:
                self.acq.set_channel_config(phys, coupling="DC", min_input=None, max_input=None)
            except Exception:
                pass

        self._building_table = False

        # Aggiorna i suffissi dei limiti per tutte le righe.  Quando la
        # tabella viene inizializzata i sensori sono impostati su "Voltage",
        # quindi non hanno unit?, ma questa chiamata predispone i controlli
        # affinch? reagiscano correttamente quando il tipo di risorsa
        # cambia.
        try:
            self._update_limit_units_all()
        except Exception:
            pass

    def _type_changed_for_row(self, row: int):
        phys = self.table.item(row, COL_PHYS).text().strip()
        cmb: QtWidgets.QComboBox = self.table.cellWidget(row, COL_TYPE)
        chosen = cmb.currentText().strip() if cmb else "Voltage"
        self._sensor_type_by_phys[phys] = chosen

        if chosen == "Voltage" or chosen == "":
            calib = {"unit":"", "a":1.0, "b":0.0}
        else:
            defs = self._read_sensor_defs()
            calib = defs.get(chosen, {"unit":"", "a":1.0, "b":0.0})

        self._calib_by_phys[phys] = calib
        self._rebuild_legends()
        self._push_sensor_map_to_core()
        # Aggiorna la configurazione del canale poich? la calibrazione potrebbe
        # influire sulla conversione dei limiti.
        try:
            self._on_config_changed_for_row(row)
        except Exception:
            pass

        # Aggiorna i suffissi dei limiti per questa riga in base alla nuova unit?.
        try:
            self._update_limit_units_for_row(row)
        except Exception:
            pass

    # ----------------------------- Config coupling/limits -----------------------------
    def _on_config_changed_for_row(self, row: int):
        """
        Called when the user changes the coupling or the physical limits for a
        given row. This method pushes the updated configuration to the core
        acquisition manager. If the table is being built or updated
        programmatically, this method does nothing.
        """
        if getattr(self, '_building_table', False):
            return
        # Determine the physical channel for this row
        try:
            item_phys = self.table.item(row, COL_PHYS)
            phys = item_phys.text().strip() if item_phys else ""
        except Exception:
            phys = ""
        if not phys:
            return
        # Read the coupling selection
        coupl = None
        try:
            w = self.table.cellWidget(row, COL_COUPLING)
            if isinstance(w, QtWidgets.QComboBox):
                coupl = w.currentText().strip()
        except Exception:
            pass
        # Read max and min limits from spin boxes
        max_v = None
        min_v = None
        try:
            wmax = self.table.cellWidget(row, COL_LIMIT_MAX)
            if isinstance(wmax, QtWidgets.QDoubleSpinBox):
                max_v = wmax.value()
        except Exception:
            pass
        try:
            wmin = self.table.cellWidget(row, COL_LIMIT_MIN)
            if isinstance(wmin, QtWidgets.QDoubleSpinBox):
                min_v = wmin.value()
        except Exception:
            pass
        # Push configuration to the core
        try:
            self.acq.set_channel_config(phys, coupling=coupl, min_input=min_v, max_input=max_v)
        except Exception:
            pass

    def _push_channel_labels_to_core(self):
        """Invia al core i nomi canale (per ogni riga della tabella)."""
        n = self.table.rowCount()
        for r in range(n):
            phys = self.table.item(r, COL_PHYS).text() if self.table.item(r, COL_PHYS) else ""
            lbl_item = self.table.item(r, COL_LABEL)
            label = lbl_item.text().strip() if lbl_item else ""
            if not label:
                label = phys
            try:
                self.acq.set_ui_name_for_phys(phys, label)
            except Exception:
                pass

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if item is None:
            return

        # evita rientri durante build/aggiornamenti programmatici
        if self._building_table or self._auto_change:
            return

        row = item.row()
        col = item.column()

        if col == COL_LABEL:
            # --- Rinominare NON deve toccare l'acquisizione ---
            phys = self.table.item(row, COL_PHYS).text().strip() if self.table.item(row, COL_PHYS) else ""
            # Etichetta digitata dall'utente (fallback al nome fisico se vuota)
            new_label = (item.text() or "").strip() or phys

            # Deduplica il nuovo nome rispetto agli altri canali.  Se esiste gi? un
            # altro canale con la stessa etichetta (ignorando la differenza
            # maiuscole/minuscole), appende un suffisso _2, _3, ... fino a trovare
            # un nome non in uso.  Questa logica evita ambiguit? quando i nomi
            # duplicati vengono usati per instradare i dati dal core alla UI.
            try:
                base = new_label
                if base:
                    # Raccogli tutte le etichette degli altri canali (case-insensitive)
                    existing = []
                    for r in range(self.table.rowCount()):
                        if r == row:
                            continue
                        it_lbl = self.table.item(r, COL_LABEL)
                        if it_lbl:
                            txt = (it_lbl.text() or "").strip()
                            if txt:
                                existing.append(txt.lower())
                    # Se il nuovo nome ? gi? presente, trova un suffisso libero
                    if base.lower() in existing:
                        suffix = 2
                        candidate = f"{base}_{suffix}"
                        while candidate.lower() in existing:
                            suffix += 1
                            candidate = f"{base}_{suffix}"
                        # Aggiorna la cella con il nome deduplicato evitando eventi di ricorsione
                        self._auto_change = True
                        item.setText(candidate)
                        self._auto_change = False
                        new_label = candidate
            except Exception:
                # In caso di errore durante la deduplicazione, continua con il nome originale
                pass

            # Aggiorna il mapping locale e le legende con l'etichetta (deduplicata)
            self._label_by_phys[phys] = new_label
            self._rebuild_legends()

            # Invia subito il nome canale deduplicato al core, in modo che i TDMS
            # utilizzino nomi univoci e coerenti.
            try:
                self.acq.set_ui_name_for_phys(phys, new_label)
            except Exception:
                pass

            # Opzionale: aggiorna l'etichetta della frequenza se l'acquisizione ? attiva.
            # Usiamo il flag interno _running invece dello stato del pulsante Stop,
            # poich? quest'ultimo viene abilitato solo durante il salvataggio.
            try:
                if getattr(self.acq, '_running', False):
                    self._update_rate_label(self._current_phys_order)
            except Exception:
                pass
            return  # <? importante: NON proseguire

        # altri casi che possono richiedere riconfigurazione
        if col == COL_ENABLE:
            self._update_acquisition_from_table()

    def _enabled_phys_and_labels(self):
        phys, labels = [], []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, COL_ENABLE)
            if it and it.checkState() == QtCore.Qt.Checked:
                p = self.table.item(r, COL_PHYS).text().strip()
                l = self.table.item(r, COL_LABEL).text().strip() or p
                phys.append(p); labels.append(l)
        return phys, labels

    def _find_row_by_phys(self, phys: str):
        for r in range(self.table.rowCount()):
            if self.table.item(r, COL_PHYS).text().strip() == phys:
                return r
        return -1

    # ----------------------------- Start/Stop auto -----------------------------
    def _update_acquisition_from_table(self):
        # Aggiorna la configurazione dei canali da coupling e limiti prima di avviare l'acquisizione
        try:
            for r in range(self.table.rowCount()):
                self._on_config_changed_for_row(r)
        except Exception:
            pass
        if not self._device_ready:
            QtWidgets.QMessageBox.warning(self, "Attenzione", "Seleziona un dispositivo prima.")
            self._auto_change = True
            for r in range(self.table.rowCount()):
                it = self.table.item(r, COL_ENABLE)
                if it: it.setCheckState(QtCore.Qt.Unchecked)
            self._auto_change = False
            self._sync_emit_status_update()
            return

        # PRENDE SEMPRE IL "NAME" PULITO dal userData della combo
        device = (self.cmbDevice.currentData() or self.cmbDevice.currentText()).strip()
        phys, labels = self._enabled_phys_and_labels()

        if not phys:
            self._stop_acquisition_ui_only()
            self.lblRateInfo.setText("-")
            self._sync_emit_status_update()
            return

        # If the set of enabled channels has not changed and an acquisition is
        # currently running, simply update the rate label and return.  We no
        # longer rely on the state of the Stop/Recompose button here because
        # that button is only enabled when recording is active, not when the
        # acquisition is running.
        if phys == self._last_enabled_phys and getattr(self.acq, '_running', False):
            self._update_rate_label(phys)
            return

        # If an acquisition is already running, stop it before starting a new
        # one with the updated list of channels.  Use the internal running flag
        # rather than the Stop/Recompose button state.
        if getattr(self.acq, '_running', False):
            try:
                self.acq.stop()
            except Exception:
                pass

        sync_cfg = dict(self._pending_sync_start_cfg or {})
        ok = self.acq.start(
            device_name=device,
            ai_channels=phys,
            channel_names=labels,
            sync_start_cfg=sync_cfg if sync_cfg else None,
        )
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Errore", "Impossibile avviare l'acquisizione.")
            return

        # Ensure that channel names used by the core are unique.  Duplicated labels
        # can cause the mapping from start names back to physical channels to be
        # ambiguous.  Use the acquisition manager helper to deduplicate labels.
        try:
            # set_channel_labels updates the internal _channel_names list with
            # unique names.  These names will be used for TDMS channels.  To
            # ensure that callbacks from the acquisition core provide these
            # deduplicated names as well, also update _start_channel_names.
            self.acq.set_channel_labels(labels)
            # Retrieve the sanitized names; fall back to the original list on error
            labels = list(self.acq._channel_names)
            # Update the start_channel_names so that callback events emit the
            # deduplicated names.  Without this assignment, the acquisition
            # core would continue to use the original (possibly duplicated)
            # names for callbacks, leading to ambiguous routing in the UI.
            try:
                self.acq._start_channel_names = labels[:]
            except Exception:
                pass
        except Exception:
            # In case of any error, keep the provided labels
            pass
        # Record the current order of physical channels and the labels used at
        # acquisition start.  These mappings are used to route incoming data
        # (start names) back to the correct physical channel.
        self._current_phys_order = phys[:]
        self._start_label_by_phys = dict(zip(phys, labels))
        self._last_enabled_phys = phys[:]

        # grafici
        self._reset_plots_curves()
        # Enable the Stop/Recompose button only when recording is active.  At this
        # point a new acquisition has just started but recording (salvataggio) has
        # not yet been enabled via the "Salva dati" button, so leave it disabled.
        try:
            self.btnStop.setEnabled(bool(self.acq.recording_enabled))
        except Exception:
            # Fallback: disable the button if we cannot read the recording state
            self.btnStop.setEnabled(False)
        if not self.guiTimer.isActive():
            self.guiTimer.start()

        self._update_rate_label(phys)
        self._push_sensor_map_to_core()
        if self._fft_enabled:
            try:
                self._emit_fft_worker_config()
            except Exception:
                pass
        self._sync_emit_status_update()

    def _update_rate_label(self, phys_list):
        try:
            labels = [self._label_by_phys.get(p, p) for p in phys_list]
            cur_per = (self.acq.current_rate_hz or 0) / 1e3
            cur_agg = (self.acq.current_agg_rate_hz or 0) / 1e3
            lim_single = (self.acq.max_single_rate_hz or 0) / 1e3
            lim_multi  = (self.acq.max_multi_rate_hz  or 0) / 1e3
            self.lblRateInfo.setText(
                f"Canali: {', '.join(labels)}  |  "
                f"Rate per-canale {cur_per:.1f} kS/s  (agg: {cur_agg:.1f} kS/s)  |  "
                f"Limiti modulo â†’ single {lim_single:.1f} kS/s, aggregato {lim_multi:.1f} kS/s"
            )
        except Exception:
            self.lblRateInfo.setText("-")

    def _on_rate_edit_finished(self):
        """
        Slot invoked when the user finishes editing the sample rate field (Fs [Hz]).
        Validates the input, applies the user-defined sampling rate to the
        acquisition manager, and restarts the acquisition if it is currently
        running. The special value "Max" (case-insensitive) or an empty field
        reverts to the automatic maximum rate.
        """
        # Read and normalize the text
        txt = (self.rateEdit.text() or "").strip()
        # Determine if the user wants the maximum rate
        use_max = False
        if txt == "" or txt.lower() == "max":
            use_max = True
        # Try to parse a numeric rate
        user_rate: Optional[float] = None
        if not use_max:
            try:
                val = float(txt)
                if val > 0:
                    user_rate = val
                else:
                    use_max = True
            except Exception:
                use_max = True
        # Apply the rate to the acquisition manager
        try:
            if use_max:
                # Use automatic maximum: reset text to "Max"
                self.rateEdit.setText("Max")
                self.acq.set_user_rate_hz(None)
            else:
                # Set the user-defined rate
                self.acq.set_user_rate_hz(user_rate)
        except Exception:
            pass
        # If an acquisition is currently running, restart it with the new sampling rate.
        # We use the internal running flag rather than the state of the Stop/Recompose
        # button because that button is only enabled when recording (salvataggio) is active.
        if getattr(self.acq, '_running', False):
            try:
                # Get current enabled channels and labels
                phys, labels = self._enabled_phys_and_labels()
                if phys:
                    # Identify the selected device
                    device = (self.cmbDevice.currentData() or self.cmbDevice.currentText()).strip()
                    # Stop the current acquisition
                    try:
                        self.acq.stop()
                    except Exception:
                        pass
                    # Restart with the same channels and labels
                    ok = False
                    try:
                        ok = self.acq.start(device_name=device, ai_channels=phys, channel_names=labels)
                    except Exception:
                        ok = False
                    if ok:
                        # Update state variables as in _update_acquisition_from_table()
                        self._current_phys_order = phys[:]
                        self._start_label_by_phys = dict(zip(phys, labels))
                        self._last_enabled_phys = phys[:]
                        # Recreate curves with distinct colours
                        self._reset_plots_curves()
                        # Enable the Stop/Recompose button only if we are currently recording.
                        try:
                            self.btnStop.setEnabled(bool(self.acq.recording_enabled))
                        except Exception:
                            self.btnStop.setEnabled(False)
                        if not self.guiTimer.isActive():
                            self.guiTimer.start()
                        # Update the rate label and push the sensor map to core
                        try:
                            self._update_rate_label(phys)
                        except Exception:
                            pass
                        try:
                            self._push_sensor_map_to_core()
                        except Exception:
                            pass
            except Exception:
                pass
        if self._fft_enabled:
            try:
                self._emit_fft_worker_config()
            except Exception:
                pass

    def _stop_acquisition_ui_only(self):
        try:
            if self.acq.recording_enabled:
                self.acq.set_recording(False)
        except Exception:
            pass
        try:
            self.acq.stop()
        except Exception:
            pass
        self.btnStop.setEnabled(False)
        if self.guiTimer.isActive():
            self.guiTimer.stop()
        self.lblRateInfo.setText("-")

    # ----------------------------- TDMS: folder/name, start/stop, countdown -----------------------------
    def _choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Seleziona cartella di salvataggio", self.txtSaveDir.text() or DEFAULT_SAVE_DIR
        )
        if folder:
            self.txtSaveDir.setText(folder)

    def _on_start_saving(self):
        # Deve essere attiva un'acquisizione per iniziare a salvare.  Usiamo lo
        # stato interno dell'acquisition manager invece del pulsante Stop, che
        # viene abilitato solo durante il salvataggio.
        try:
            is_running = bool(getattr(self.acq, '_running', False))
        except Exception:
            is_running = False
        if not is_running:
            QtWidgets.QMessageBox.warning(self, "Attenzione", "Abilita almeno un canale per avviare il salvataggio.")
            return

        base_dir = self.txtSaveDir.text().strip() or DEFAULT_SAVE_DIR
        os.makedirs(base_dir, exist_ok=True)
        base_name = self.txtBaseName.text().strip() or "SenzaNome.tdms"
        if not base_name.lower().endswith(".tdms"):
            base_name += ".tdms"

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir_name = f"{os.path.splitext(base_name)[0]}_{ts}"
        subdir = os.path.join(base_dir, subdir_name)
        os.makedirs(subdir, exist_ok=True)

        # invia i nomi canale al core (per ogni riga della tabella)
        # send channel labels and configure base filename for TDMS segments
        self._push_channel_labels_to_core()
        # imposta il limite di memoria in base al valore selezionato nella spinbox
        try:
            mem_mb = self.spinRam.value()
            if hasattr(self.acq, "set_memory_limit_mb"):
                self.acq.set_memory_limit_mb(mem_mb)
        except Exception:
            pass
        # reset any residual in-memory blocks before changing the output directory
        try:
            if hasattr(self.acq, "clear_memory_buffer"):
                self.acq.clear_memory_buffer()
        except Exception:
            pass
        # prepare a fresh output directory
        self.acq.set_output_directory(subdir)
        # set base filename (without extension) for naming the TDMS segments
        self.acq.set_base_filename(base_name)
        # enable recording so the writer will start accumulating blocks in RAM
        self.acq.set_recording(True)
        # Mark the FFT baseline for this recording and clear any stale spectrum
        # from previous runs.
        self._fft_result_counter_at_record_start = int(self._fft_result_counter)
        self._last_fft_freq = None
        self._last_fft_mag_by_phys.clear()

        # Now that recording is active, the Stop/Recompose button can be used to
        # stop and merge the temporary TDMS files.  Enable it explicitly.
        try:
            self.btnStop.setEnabled(True)
        except Exception:
            pass

        self._active_subdir = subdir
        self._save_dir = base_dir
        self._base_filename = base_name

        self._set_table_lock(True)

        # Reset and start memory usage timer for updating the save button text
        if not self._count_timer.isActive():
            self._count_timer.start()
        # Immediately update the button text with memory usage
        try:
            used, limit = self.acq.get_memory_usage()
            mb = used / (1024 * 1024)
            total_mb = limit / (1024 * 1024)
            self.btnStart.setText(f"Salvataggio ({mb:.1f} / {total_mb:.0f} MB)")
        except Exception:
            self.btnStart.setText("Salvataggio (0 / 500 MB)")

        # blocca i campi
        self.txtSaveDir.setEnabled(False)
        self.btnBrowseDir.setEnabled(False)
        self.txtBaseName.setEnabled(False)
        self.btnStart.setEnabled(False)

    def _tick_countdown(self):
        # Update the save button text with current memory usage while recording
        if not self.acq.recording_enabled:
            self._count_timer.stop()
            self.btnStart.setText("Salva dati")
            self.btnStart.setEnabled(True)
            return
        # Query memory usage from the acquisition manager
        try:
            used_bytes, limit_bytes = self.acq.get_memory_usage()
            used_mb = used_bytes / (1024 * 1024)
            total_mb = limit_bytes / (1024 * 1024)
            self.btnStart.setText(f"Salvataggio ({used_mb:.1f} / {total_mb:.0f} MB)")
        except Exception:
            self.btnStart.setText("Salvataggio")

    def _update_start_button_text(self):
        self.btnStart.setText(f"Salvo in ({self._countdown:02d} s) ...")

    def _on_stop(self):
        # ferma core
        try:
            self.acq.stop_graceful()
        except Exception:
            pass
        try:
            self.acq.stop()
        except Exception:
            pass
        # Disable FFT on stop without re-entering toggle slot logic.
        try:
            self._auto_fft_change = True
            self.chkFftEnable.setChecked(False)
        except Exception:
            pass
        finally:
            self._auto_fft_change = False
        self._fft_enabled = False
        self._clear_fft_plot_visuals(clear_cached=False)

        # cleanup temporary FFT chunk files written on disk (worker thread)
        try:
            self.sigFftWorkerReset.emit(True, True)
        except Exception:
            pass

        if self._count_timer.isActive():
            self._count_timer.stop()
        self.btnStart.setText("Salva dati")
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.guiTimer.stop()

        # riabilita campi
        self.txtSaveDir.setEnabled(True)
        self.btnBrowseDir.setEnabled(True)
        self.txtBaseName.setEnabled(True)

        # merge
        if not self._active_subdir:
            QtWidgets.QMessageBox.information(self, "Info", "Acquisizione fermata. Nessuna cartella di salvataggio attiva.")
            return

        # Determine the desired final TDMS path.  If a file with the same
        # name already exists in the save directory, append the current date and
        # time (dd_mm_yy_hh_mm_ss) to avoid overwriting it.
        final_path = os.path.join(self._save_dir, self._base_filename)
        try:
            if os.path.exists(final_path):
                # Split the base filename into name and extension
                base_name, ext = os.path.splitext(self._base_filename)
                # Current date/time string
                dt_str = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
                # Compose a new filename with the date/time appended
                new_name = f"{base_name}_{dt_str}{ext or '.tdms'}"
                final_path = os.path.join(self._save_dir, new_name)
        except Exception:
            pass
        # Merge the temporary TDMS files into the final file with progress feedback
        try:
            from tdms_merge import TdmsMerger
            merger = TdmsMerger()
            try:
                anchor = getattr(self.acq, "_sync_anchor_epoch_s", None)
                cutoff = int(getattr(self.acq, "_sync_write_start_index", 0) or 0)
                fs = float(getattr(self.acq, "current_rate_hz", 0.0) or 0.0)
                if anchor is not None:
                    forced = float(anchor)
                    if cutoff > 0 and fs > 0:
                        forced += float(cutoff) / float(fs)
                    merger.forced_wf_start_time = datetime.datetime.fromtimestamp(forced)
            except Exception:
                pass

            # -----------------------------------------------------------
            # Prepara la struttura dati FFT per l'eventuale salvataggio.
            # Se ? stata calcolata una FFT e l'utente ha richiesto la
            # visualizzazione dello spettro, il dizionario ``fft_data`` viene
            # popolato con la frequenza e le magnitudini.  Questo dizionario
            # sar? quindi assegnato all'oggetto TdmsMerger, che provveder? ad
            # appendere un segmento FFT al file unito.  In assenza di dati
            # validi, ``fft_data`` rester? None e l'oggetto merger non
            # aggiunger? il segmento.
            try:
                freq = getattr(self, '_last_fft_freq', None)
                mags = getattr(self, '_last_fft_mag_by_phys', None)
                fresh_fft_for_recording = (
                    int(getattr(self, "_fft_result_counter", 0))
                    > int(getattr(self, "_fft_result_counter_at_record_start", 0))
                )
                if fresh_fft_for_recording and isinstance(freq, np.ndarray) and freq.size > 0 and isinstance(mags, dict) and mags:
                    # Costruisci dizionario canali {nome -> array} e unit?
                    ch_map = {}
                    units_map = {}
                    for phys, arr in mags.items():
                        try:
                            if not (isinstance(arr, np.ndarray) and arr.size == freq.size):
                                continue
                        except Exception:
                            continue
                        # Usa l'etichetta di partenza (univoca) per il nome canale FFT
                        label = self._start_label_by_phys.get(phys, self._label_by_phys.get(phys, phys))
                        # Prefix FFT_ per identificare i canali spettro
                        fft_label = f"FFT_{label}"
                        ch_map[fft_label] = arr.astype(np.float64)
                        # Recupera l'unit? ingegneristica associata a questo canale
                        unit = self._calib_by_phys.get(phys, {}).get("unit", "")
                        units_map[fft_label] = unit or ""
                    if ch_map:
                        fft_data = {
                            "freq": freq.astype(np.float64),
                            "channels": ch_map,
                            "units": units_map,
                            "duration": float(getattr(self, '_fft_duration_seconds', 0))
                        }
                        merger.fft_data = fft_data
            except Exception:
                # In caso di errore nella preparazione della FFT, ignora e
                # procede con il merge dei soli dati temporali.
                pass
            # Progress dialog
            dlg = QtWidgets.QProgressDialog("Unione file TDMS in corso...", "Annulla", 0, 1, self)
            dlg.setWindowTitle("Unione in corso")
            dlg.setWindowModality(QtCore.Qt.WindowModal)
            dlg.setValue(0)
            # memorizza la cartella temporanea perch? _active_subdir verr? azzerata
            tmp_subdir = self._active_subdir
            # Define progress callback
            def _merge_progress(curr: int, total: int):
                try:
                    dlg.setMaximum(total)
                    dlg.setValue(curr)
                    QtWidgets.QApplication.processEvents()
                    if dlg.wasCanceled():
                        raise RuntimeError("Operazione di unione annullata dall'utente.")
                except Exception:
                    pass
            # Perform merge with progress callback
            merger.merge_temp_tdms(tmp_subdir, final_path, progress_cb=_merge_progress)
            dlg.close()
            QtWidgets.QMessageBox.information(self, "Completato", f"TDMS finale creato:\n{final_path}")
            # Una volta uniti i segmenti, elimina la cartella temporanea
            try:
                if tmp_subdir and os.path.isdir(tmp_subdir):
                    shutil.rmtree(tmp_subdir, ignore_errors=True)
            except Exception:
                pass
        except Exception as e:
            try:
                dlg.close()
            except Exception:
                pass
            QtWidgets.QMessageBox.critical(self, "Errore ricomposizione", str(e))
        finally:
            self._active_subdir = None
            try:
                self.acq.set_sync_write_start_index(0)
                self.acq.set_sync_anchor_epoch_s(None)
            except Exception:
                pass

        self._set_table_lock(False)
        self._uncheck_all_enabled()

    # ----------------------------- Grafici -----------------------------
    def _get_fft_chunk_dir(self) -> str:
        base_dir = self.txtSaveDir.text().strip() if hasattr(self, "txtSaveDir") else ""
        if not base_dir:
            base_dir = self._save_dir or DEFAULT_SAVE_DIR
        return os.path.join(base_dir, "_fft_chunks")

    def _reset_fft_disk_state(self, cleanup_files: bool, cleanup_temp_npz: bool = True) -> None:
        if cleanup_files:
            d = getattr(self, "_fft_chunk_dir", "") or self._get_fft_chunk_dir()
            if d and os.path.isdir(d):
                try:
                    for p in glob.glob(os.path.join(d, "fft_window_*")):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                    if cleanup_temp_npz:
                        for p in glob.glob(os.path.join(d, "fft_temp_window_*.npz")):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                        # Remove the temp directory itself on full cleanup.
                        try:
                            shutil.rmtree(d, ignore_errors=True)
                        except Exception:
                            pass
                except Exception:
                    pass
        self._fft_chunk_samples = 0
        self._fft_chunk_fs_est = 0.0
        self._fft_window_start_t = None
        self._fft_window_end_t = None
        self._fft_window_t_path = ""
        self._fft_window_y_paths = {}
        if cleanup_files and cleanup_temp_npz:
            self._fft_chunk_dir = ""
        if cleanup_files and cleanup_temp_npz:
            self._fft_temp_npz_files = []

    def _trim_fft_temp_npz_files(self) -> None:
        keep_n = max(1, int(getattr(self, "_fft_temp_keep_count", 3)))
        files = [p for p in list(getattr(self, "_fft_temp_npz_files", [])) if isinstance(p, str) and p]
        while len(files) > keep_n:
            old = files.pop(0)
            try:
                if os.path.isfile(old):
                    os.remove(old)
            except Exception:
                pass
        self._fft_temp_npz_files = files

    def _ensure_fft_chunk_dir(self) -> None:
        d = self._get_fft_chunk_dir()
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            return
        self._fft_chunk_dir = d

    def _open_new_fft_window_files(self) -> bool:
        self._ensure_fft_chunk_dir()
        if not self._fft_chunk_dir:
            return False
        win = int(getattr(self, "_fft_chunk_window_idx", 0))
        self._fft_window_t_path = os.path.join(self._fft_chunk_dir, f"fft_window_{win:06d}_t.f64.bin")
        self._fft_window_y_paths = {
            phys: os.path.join(self._fft_chunk_dir, f"fft_window_{win:06d}_{phys}.f32.bin")
            for phys in self._current_phys_order
        }
        return True

    def _append_fft_chunk_to_disk(self, t: np.ndarray, eng_by_phys: Dict[str, np.ndarray]) -> None:
        if not self._fft_enabled:
            return
        if not isinstance(t, np.ndarray) or t.size < 2:
            return
        if not self._fft_window_t_path:
            if not self._open_new_fft_window_files():
                return

        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.size < 2:
            return

        fs_est = self._estimate_fs_from_time_vector(t_arr)
        if fs_est <= 0.0:
            try:
                fs_est = float(getattr(self.acq, "current_rate_hz", 0.0) or 0.0)
            except Exception:
                fs_est = 0.0
        if fs_est > 0.0:
            self._fft_chunk_fs_est = fs_est

        try:
            with open(self._fft_window_t_path, "ab") as f:
                t_arr.tofile(f)
        except Exception:
            return

        for phys in self._current_phys_order:
            y = eng_by_phys.get(phys)
            if isinstance(y, np.ndarray) and y.size == t_arr.size:
                y_arr = np.asarray(y, dtype=np.float32)
            else:
                y_arr = np.full(int(t_arr.size), np.nan, dtype=np.float32)
            p = self._fft_window_y_paths.get(phys, "")
            if not p:
                continue
            try:
                with open(p, "ab") as f:
                    y_arr.tofile(f)
            except Exception:
                pass

        self._fft_chunk_samples += int(t_arr.size)
        t0 = float(t_arr[0])
        t1 = float(t_arr[-1])
        if self._fft_window_start_t is None:
            self._fft_window_start_t = t0
        self._fft_window_end_t = t1

    def _build_fft_window_npz_and_compute(self) -> None:
        if self._fft_chunk_samples < 2:
            return
        t0 = self._fft_window_start_t
        t1 = self._fft_window_end_t
        if t0 is None or t1 is None:
            return
        elapsed = max(0.0, float(t1) - float(t0))
        target = max(1.0, float(self._fft_duration_seconds))
        fmax = max(0.0, float(self._fft_chunk_fs_est or 0.0) / 2.0)
        try:
            self.lblFftStatus.setText(f"FFT: finestra su disco {elapsed:.2f}/{target:g}s - F max= {fmax:.3g} Hz")
        except Exception:
            pass
        if elapsed < target:
            return

        self._ensure_fft_chunk_dir()
        if not self._fft_chunk_dir:
            return
        final_npz = os.path.join(
            self._fft_chunk_dir,
            f"fft_temp_window_{int(self._fft_chunk_window_idx):06d}.npz",
        )

        if not self._fft_window_t_path or not os.path.isfile(self._fft_window_t_path):
            self._reset_fft_disk_state(cleanup_files=True, cleanup_temp_npz=False)
            self._fft_chunk_window_idx += 1
            return
        t_all = np.fromfile(self._fft_window_t_path, dtype=np.float64)
        if t_all.size < 64:
            self._reset_fft_disk_state(cleanup_files=True, cleanup_temp_npz=False)
            self._fft_chunk_window_idx += 1
            return

        y_all = {}
        n = int(t_all.size)
        for phys in self._current_phys_order:
            p = self._fft_window_y_paths.get(phys, "")
            if p and os.path.isfile(p):
                y = np.fromfile(p, dtype=np.float32)
                if y.size < n:
                    pad = np.full(n - y.size, np.nan, dtype=np.float32)
                    y = np.concatenate([y, pad])
                elif y.size > n:
                    y = y[:n]
            else:
                y = np.full(n, np.nan, dtype=np.float32)
            y_all[phys] = y

        out = {"t": t_all, "phys": np.asarray(self._current_phys_order, dtype=object)}
        for phys in self._current_phys_order:
            out[f"y_{phys}"] = y_all[phys]
        np.savez(final_npz, **out)
        self._fft_temp_npz_files.append(final_npz)
        self._trim_fft_temp_npz_files()

        self._compute_fft_from_npz(final_npz, fs_hint=float(self._fft_chunk_fs_est or 0.0))

        self._reset_fft_disk_state(cleanup_files=True, cleanup_temp_npz=False)
        self._fft_chunk_window_idx += 1

    def _compute_fft_from_npz(self, npz_path: str, fs_hint: float = 0.0) -> None:
        try:
            with np.load(npz_path, allow_pickle=True) as z:
                t_all = np.asarray(z["t"], dtype=np.float64)
                if t_all.size < 64:
                    return
                fs = self._estimate_fs_from_time_vector(t_all)
                if fs <= 0.0:
                    fs = float(fs_hint or 0.0)
                if fs <= 0.0:
                    return

                self._last_fft_freq = None
                self._last_fft_mag_by_phys.clear()
                peak_strings = []

                for phys in self._current_phys_order:
                    key = f"y_{phys}"
                    if key not in z:
                        continue
                    y = np.asarray(z[key], dtype=np.float64)
                    if y.size != t_all.size or y.size < 64:
                        continue
                    if not np.all(np.isfinite(y)):
                        valid = np.isfinite(y)
                        if np.count_nonzero(valid) < 64:
                            continue
                        y = np.interp(t_all, t_all[valid], y[valid])
                    freq, mag = self._compute_welch_magnitude(y, fs)
                    if not (isinstance(freq, np.ndarray) and isinstance(mag, np.ndarray)):
                        continue
                    if self._last_fft_freq is None:
                        self._last_fft_freq = freq
                    if self._last_fft_freq is None or mag.size != self._last_fft_freq.size:
                        continue
                    self._last_fft_mag_by_phys[phys] = mag
                    curve = self._fft_curves_by_phys.get(phys)
                    if curve is not None:
                        try:
                            curve.setData(self._last_fft_freq, mag)
                        except Exception:
                            pass
                    if mag.size > 1:
                        idx_peak = int(np.argmax(mag[1:])) + 1
                        amp_peak = float(mag[idx_peak])
                        f_peak = float(self._last_fft_freq[idx_peak])
                        label = self._start_label_by_phys.get(phys, self._label_by_phys.get(phys, phys))
                        unit = self._calib_by_phys.get(phys, {}).get("unit", "")
                        peak_strings.append(f"{label}: {amp_peak:.3g}{(' ' + unit) if unit else ''} @ {f_peak:.6g} Hz")

                try:
                    self.lblFftPeakInfo.setText("; ".join(peak_strings))
                except Exception:
                    pass
        except Exception:
            return

    def _get_effective_fft_target_fs(self, fs_in: float = 0.0) -> float:
        """
        Return adaptive target Fs for FFT buffering.
        Constrained by requested target, input Fs, duration and max samples.
        """
        try:
            base_target = float(self._fft_target_fs_hz)
        except Exception:
            base_target = 1000.0
        base_target = max(10.0, base_target)
        try:
            dur = float(self._fft_duration_seconds)
        except Exception:
            dur = 5.0
        dur = max(1.0, dur)
        try:
            max_samp = int(self._fft_max_samples)
        except Exception:
            max_samp = 200_000
        max_samp = max(2048, max_samp)
        by_mem = max(10.0, float(max_samp) / dur)
        tgt = min(base_target, by_mem)
        if fs_in > 0.0:
            tgt = min(tgt, fs_in)
        return max(10.0, tgt)

    def _reset_fft_buffers(self):
        """
        Recreate dedicated FFT ring buffers using current duration and channels.
        Keeps memory bounded and independent from chart buffers.
        """
        try:
            fs_in = float(getattr(self.acq, "current_rate_hz", 0.0) or 0.0)
        except Exception:
            fs_in = 0.0
        target_fs = self._get_effective_fft_target_fs(fs_in=fs_in)
        try:
            req_s = float(self._fft_duration_seconds) * 1.15 + 1.0
        except Exception:
            req_s = 5.0
        maxlen = max(256, int(round(req_s * target_fs)))
        try:
            maxlen = min(maxlen, int(self._fft_max_samples))
        except Exception:
            pass

        try:
            old_t = list(self._fft_t)
        except Exception:
            old_t = []
        self._fft_t = deque(old_t[-maxlen:], maxlen=maxlen)

        old_map = dict(getattr(self, "_fft_y_by_phys", {}) or {})
        active_phys = list(self._current_phys_order) if self._current_phys_order else list(old_map.keys())
        if not active_phys:
            try:
                n = int(getattr(self.acq, "num_channels", 4) or 4)
            except Exception:
                n = 4
            active_phys = [f"ai{i}" for i in range(max(1, n))]

        new_map = {}
        for phys in active_phys:
            old_dq = old_map.get(phys, [])
            new_map[phys] = deque(list(old_dq)[-maxlen:], maxlen=maxlen)
        self._fft_y_by_phys = new_map

    def _estimate_fs_from_time_vector(self, t: np.ndarray) -> float:
        """Estimate sampling frequency from a monotonic time vector."""
        if not isinstance(t, np.ndarray) or t.size < 2:
            return 0.0
        try:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0.0)]
            if dt.size < 1:
                return 0.0
            fs = 1.0 / float(np.median(dt))
            return fs if np.isfinite(fs) and fs > 0.0 else 0.0
        except Exception:
            return 0.0

    def _compute_welch_magnitude(self, x: np.ndarray, fs: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Magnitude spectrum from a long FFT window.
        Uses a power-of-two tail to keep computation efficient while preserving
        low-frequency resolution for long acquisitions.
        """
        if not isinstance(x, np.ndarray):
            return None, None
        n = int(x.size)
        if n < 64 or fs <= 0.0:
            return None, None

        try:
            nfft_cap = int(self._fft_max_samples)
        except Exception:
            nfft_cap = 200_000
        n_use = min(n, max(64, nfft_cap))
        nfft = 1 << int(np.floor(np.log2(max(64, n_use))))
        if nfft < 64:
            return None, None

        seg = np.asarray(x[-nfft:], dtype=np.float64, copy=False)
        seg = seg - float(np.mean(seg))
        win = np.hanning(nfft).astype(np.float64, copy=False)
        w_sum = float(np.sum(win))
        if w_sum <= 0.0:
            return None, None
        spec = np.fft.rfft(seg * win)
        mag = (2.0 / w_sum) * np.abs(spec)
        freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
        return freq, mag

    def _reset_plots(self):
        self._chart_x.clear()
        for dq in self._chart_y_by_phys.values(): dq.clear()
        self._instant_t = np.array([], dtype=float)
        self._instant_y_by_phys = {k: np.array([], dtype=float) for k in self._instant_y_by_phys}
        self._reset_fft_buffers()
        self._last_fft_compute_ts = 0.0

        for c in list(self._chart_curves_by_phys.values()):
            try: self.pgChart.removeItem(c)
            except Exception: pass
        for c in list(self._instant_curves_by_phys.values()):
            try: self.pgInstant.removeItem(c)
            except Exception: pass
        self._chart_curves_by_phys.clear()
        self._instant_curves_by_phys.clear()
        # Rimuove eventuali curve FFT e pulisce la legenda FFT
        for c in list(getattr(self, '_fft_curves_by_phys', {}).values()):
            try:
                self.pgFFT.removeItem(c)
            except Exception:
                pass
        try:
            if hasattr(self, '_fft_curves_by_phys'):
                self._fft_curves_by_phys.clear()
        except Exception:
            pass
        try:
            if hasattr(self, '_fft_legend'):
                self._fft_legend.clear()
        except Exception:
            pass
        self._chart_legend.clear()
        self._instant_legend.clear()

        # Cancella la stringa delle medie quando si resettano i grafici
        try:
            if hasattr(self, 'lblAvgChart'):
                self.lblAvgChart.setText("")
        except Exception:
            pass

    def _reset_plots_curves(self):
        self._reset_plots()
        # Assign distinct colors to each channel for better readability
        num_ch = max(1, len(self._current_phys_order))
        for idx, phys in enumerate(self._current_phys_order):
            unit = self._calib_by_phys.get(phys, {}).get("unit", "")
            base_label = self._label_by_phys.get(phys, phys)
            label = f"{base_label} [{unit}]" if unit else base_label
            # Use a distinct color based on the index; pg.intColor returns a QColor
            try:
                color = pg.intColor(idx, hues=max(8, num_ch))
            except Exception:
                color = (255, 0, 0)  # fallback to red
            # Chart (concatenated)
            c = self.pgChart.plot(name=label, pen=pg.mkPen(color=color, width=1.5))
            try:
                c.setClipToView(True)
                c.setDownsampling(auto=True, mode='peak')
            except Exception:
                pass
            self._chart_curves_by_phys[phys] = c
            self._chart_legend.addItem(c, label)
            # Instant block plot
            ic = self.pgInstant.plot(name=label, pen=pg.mkPen(color=color, width=1.5))
            try:
                ic.setClipToView(True)
                ic.setDownsampling(auto=True, mode='peak')
            except Exception:
                pass
            self._instant_curves_by_phys[phys] = ic
            self._instant_legend.addItem(ic, label)

            # FFT plot: crea una curva per questo canale e aggiungila alla
            # legenda FFT.  Non si applica decimazione o clipping qui perch?
            # l'FFT ? gi? un riassunto della finestra selezionata.
            try:
                cfft = self.pgFFT.plot(name=label, pen=pg.mkPen(color=color, width=1.5))
            except Exception:
                cfft = None
            if cfft is not None:
                try:
                    # Lascio ClipToView disabilitato per l'FFT; eventuale
                    # riduzione di punti sar? gestita dal calcolo stesso.
                    pass
                except Exception:
                    pass
                self._fft_curves_by_phys[phys] = cfft
                try:
                    self._fft_legend.addItem(cfft, label)
                except Exception:
                    pass

        # Reset last FFT results and title when channels are rebuilt.
        self._last_fft_freq = None
        self._last_fft_mag_by_phys.clear()
        try:
            title = f"Spettro FFT (finestra {self._fft_duration_seconds:.3g} s)"
            self.pgFFT.setTitle(title)
        except Exception:
            pass

    def _rebuild_legends(self):
        self._chart_legend.clear()
        self._instant_legend.clear()
        for phys, curve in self._chart_curves_by_phys.items():
            unit = self._calib_by_phys.get(phys, {}).get("unit", "")
            base_label = self._label_by_phys.get(phys, phys)
            label = f"{base_label} [{unit}]" if unit else base_label
            try: curve.setName(label)
            except Exception: pass
            self._chart_legend.addItem(curve, label)
        for phys, curve in self._instant_curves_by_phys.items():
            unit = self._calib_by_phys.get(phys, {}).get("unit", "")
            base_label = self._label_by_phys.get(phys, phys)
            label = f"{base_label} [{unit}]" if unit else base_label
            try: curve.setName(label)
            except Exception: pass
            self._instant_legend.addItem(curve, label)

        # FFT legend: aggiorna il nome e la legenda per ogni canale
        try:
            if hasattr(self, '_fft_legend'):
                self._fft_legend.clear()
        except Exception:
            pass
        for phys, curve in getattr(self, '_fft_curves_by_phys', {}).items():
            unit = self._calib_by_phys.get(phys, {}).get("unit", "")
            base_label = self._label_by_phys.get(phys, phys)
            label = f"{base_label} [{unit}]" if unit else base_label
            try:
                curve.setName(label)
            except Exception:
                pass
            try:
                self._fft_legend.addItem(curve, label)
            except Exception:
                pass

    def _clear_chart(self):
        self._chart_x.clear()
        for phys, curve in self._chart_curves_by_phys.items():
            self._chart_y_by_phys[phys].clear()
            curve.clear()

    # slot dai segnali (main thread)
    def _slot_instant_block(self, t: np.ndarray, ys: list, names: list):
        try:
            self._instant_t = np.asarray(t, dtype=float)
            # mappa nome di start -> phys
            start_to_phys = {name: phys for phys, name in self._start_label_by_phys.items()}
            eng_by_phys = {}
            for name, y in zip(names, ys):
                phys = start_to_phys.get(name)
                if not phys: continue
                cal = self._calib_by_phys.get(phys, {"a":1.0,"b":0.0})
                a = float(cal.get("a",1.0)); b = float(cal.get("b",0.0))
                y_eng = a * np.asarray(y, dtype=float) + b
                self._instant_y_by_phys[phys] = y_eng
                eng_by_phys[phys] = y_eng
            if self._fft_enabled:
                payload = {"t": np.asarray(self._instant_t, dtype=np.float64), "y_map": eng_by_phys}
                self.sigFftWorkerBlock.emit(payload)

        except Exception:
            try:
                self.lblFftStatus.setText("FFT: errore invio blocco al worker")
            except Exception:
                pass
            pass

    def _slot_chart_points(self, t_pts: np.ndarray, ys_pts: list, names: list):
        try:
            t_pts = np.asarray(t_pts, dtype=float)
            self._chart_x.extend(t_pts.tolist())
            start_to_phys = {name: phys for phys, name in self._start_label_by_phys.items()}
            for name, ypts in zip(names, ys_pts):
                phys = start_to_phys.get(name)
                if not phys: continue
                cal = self._calib_by_phys.get(phys, {"a":1.0,"b":0.0})
                a = float(cal.get("a",1.0)); b = float(cal.get("b",0.0))
                y_eng = a * np.asarray(ypts, dtype=float) + b
                self._chart_y_by_phys[phys].extend(y_eng.tolist())
        except RuntimeError:
            pass

    def _on_instant_view_toggled(self, checked: bool) -> None:
        self._instant_plot_enabled = bool(checked)
        if self._instant_plot_enabled:
            return
        for curve in self._instant_curves_by_phys.values():
            try:
                curve.clear()
            except Exception:
                pass

    def _refresh_plots(self):
        # chart concatenato
        n = len(self._chart_x)
        if n > 1:
            x = np.fromiter(self._chart_x, dtype=float, count=n)
            for phys, curve in self._chart_curves_by_phys.items():
                dq = self._chart_y_by_phys.get(phys)
                if not dq:
                    continue
                y = np.fromiter(dq, dtype=float, count=len(dq))
                if y.size == x.size and y.size > 1:
                    curve.setData(x, y)

        # blocco istantaneo
        if self._instant_plot_enabled and isinstance(self._instant_t, np.ndarray) and self._instant_t.size > 1:
            for phys, curve in self._instant_curves_by_phys.items():
                y = self._instant_y_by_phys.get(phys)
                if isinstance(y, np.ndarray) and y.size == self._instant_t.size and y.size > 1:
                    curve.setData(self._instant_t, y)

        # Calcola il valore medio per ogni canale attivo e aggiorna l'etichetta
        try:
            if hasattr(self, 'lblAvgChart'):
                avg_strings = []
                for phys in self._current_phys_order:
                    dq = self._chart_y_by_phys.get(phys)
                    if dq and len(dq) > 0:
                        try:
                            y_vals = np.fromiter(dq, dtype=float, count=len(dq))
                            if y_vals.size > 0:
                                avg_val = float(np.mean(y_vals))
                                label = self._start_label_by_phys.get(phys, self._label_by_phys.get(phys, phys))
                                unit = self._calib_by_phys.get(phys, {}).get('unit', '')
                                avg_strings.append(f"{label}: {avg_val:.6g}" + (f" {unit}" if unit else ""))
                        except Exception:
                            pass
                self.lblAvgChart.setText(", ".join(avg_strings) if avg_strings else "")
        except Exception:
            pass

        # FFT computation is disk-driven in _slot_instant_block.

    def _set_fft_window_lines_visible(self, visible: bool) -> None:
        # Window markers over decimated chart intentionally disabled.
        return

    def _update_fft_window_lines(self) -> None:
        # Window markers over decimated chart intentionally disabled.
        return

    def _apply_fft_axis_limits(self) -> None:
        try:
            xmin = float(self.spinFftXMin.value())
            xmax = float(self.spinFftXMax.value())
            ymin = float(self.spinFftYMin.value())
            ymax = float(self.spinFftYMax.value())
        except Exception:
            return

        if not (xmax > xmin and ymax > ymin):
            QtWidgets.QMessageBox.warning(self, "Assi FFT", "Intervalli non validi: serve max > min.")
            return
        if self.chkFftLogScale.isChecked() and (xmin <= 0.0 or xmax <= 0.0 or ymin <= 0.0 or ymax <= 0.0):
            QtWidgets.QMessageBox.warning(self, "Assi FFT", "In scala log-log gli estremi devono essere > 0.")
            return
        try:
            if self.chkFftLogScale.isChecked():
                # In pyqtgraph log-mode the view range is in log10 coordinates.
                self.pgFFT.setXRange(np.log10(xmin), np.log10(xmax), padding=0.0)
                self.pgFFT.setYRange(np.log10(ymin), np.log10(ymax), padding=0.0)
            else:
                self.pgFFT.setXRange(xmin, xmax, padding=0.0)
                self.pgFFT.setYRange(ymin, ymax, padding=0.0)
        except Exception:
            pass

    def _auto_fft_axis_limits(self) -> None:
        try:
            self.pgFFT.enableAutoRange(axis='xy', enable=True)
            self.pgFFT.getPlotItem().autoRange()
        except Exception:
            pass

    @QtCore.pyqtSlot(str)
    def _on_fft_worker_status(self, text: str) -> None:
        try:
            self.lblFftStatus.setText(str(text or ""))
        except Exception:
            pass

    @QtCore.pyqtSlot(object, object, object)
    def _on_fft_worker_result(self, freq: object, mag_map: object, _peak_text: object) -> None:
        if not isinstance(freq, np.ndarray):
            return
        if not isinstance(mag_map, dict):
            return
        self._last_fft_freq = np.asarray(freq, dtype=np.float64)
        self._last_fft_mag_by_phys.clear()
        peak_strings = []
        for phys, mag in mag_map.items():
            try:
                arr = np.asarray(mag, dtype=np.float64)
            except Exception:
                continue
            if arr.size != self._last_fft_freq.size:
                continue
            self._last_fft_mag_by_phys[phys] = arr
            curve = self._fft_curves_by_phys.get(phys)
            if curve is not None:
                try:
                    curve.setData(self._last_fft_freq, arr)
                except Exception:
                    pass
            if arr.size > 1:
                idx_peak = int(np.argmax(arr[1:])) + 1
                amp_peak = float(arr[idx_peak])
                f_peak = float(self._last_fft_freq[idx_peak])
                label = self._start_label_by_phys.get(phys, self._label_by_phys.get(phys, phys))
                unit = self._calib_by_phys.get(phys, {}).get('unit', '')
                peak_strings.append(f"{label}: {amp_peak:.3g}{(' ' + unit) if unit else ''} @ {f_peak:.6g} Hz")
        if self._last_fft_mag_by_phys:
            self._fft_result_counter += 1
        for phys, curve in self._fft_curves_by_phys.items():
            if phys not in self._last_fft_mag_by_phys:
                try:
                    curve.clear()
                except Exception:
                    pass
        try:
            self.lblFftPeakInfo.setText("; ".join(peak_strings))
        except Exception:
            pass

    @QtCore.pyqtSlot(str)
    def _on_fft_worker_guardrail(self, msg: str) -> None:
        try:
            self.statusBar.showMessage(str(msg), 6000)
        except Exception:
            pass
        try:
            self.lblFftStatus.setText(str(msg))
        except Exception:
            pass

    def _emit_fft_worker_config(self) -> None:
        try:
            fs = float(getattr(self.acq, "current_rate_hz", 0.0) or 0.0)
        except Exception:
            fs = 0.0
        cfg = {
            "duration_s": float(self._fft_duration_seconds),
            "save_dir": self.txtSaveDir.text().strip() or self._save_dir or DEFAULT_SAVE_DIR,
            "phys_order": list(self._current_phys_order),
            "temp_keep_count": int(getattr(self, "_fft_temp_keep_count", 3)),
            "fs_hz": fs,
        }
        self.sigFftWorkerConfig.emit(cfg)

    def _on_fft_duration_changed(self, value: float) -> None:
        """
        Slot called when the user changes FFT duration.
        """
        try:
            self._fft_duration_seconds = float(value)
        except Exception:
            self._fft_duration_seconds = 1.0
        try:
            title = f"Spettro FFT (finestra {self._fft_duration_seconds:.3g} s)"
            self.pgFFT.setTitle(title)
        except Exception:
            pass
        self._reset_fft_buffers()
        self.sigFftWorkerReset.emit(True, True)
        self._fft_chunk_window_idx = 0
        self._last_fft_compute_ts = 0.0
        self._fft_last_window_end_t = None
        self._emit_fft_worker_config()
        if self._fft_enabled:
            try:
                tgt = max(1.0, float(self._fft_duration_seconds))
                self.lblFftStatus.setText(f"FFT: finestra su disco 0.00/{tgt:g}s - F max= 0 Hz")
            except Exception:
                pass

    def _on_fft_log_scale_changed(self, checked: bool) -> None:
        """
        Slot called when the user toggles the log-scale checkbox.  Sets the
        FFT plot axes to log-log mode if checked, otherwise linear.
        """
        try:
            # PyQtGraph allows separate log settings for x and y axes.  Set
            # both to the same state to achieve a log-log or linear-linear
            # display.  When switching back to linear, any non-positive
            # samples will be handled gracefully by pyqtgraph.
            self.pgFFT.setLogMode(checked, checked)
        except Exception:
            pass

    def _clear_fft_plot_visuals(self, clear_cached: bool = True) -> None:
        for curve in self._fft_curves_by_phys.values():
            try:
                curve.clear()
            except Exception:
                pass
        try:
            self.lblFftStatus.setText("")
        except Exception:
            pass
        try:
            self.lblFftPeakInfo.setText("")
        except Exception:
            pass
        if clear_cached:
            self._last_fft_freq = None
            self._last_fft_mag_by_phys.clear()

    def _on_fft_enable_toggled(self, checked: bool) -> None:
        """
        Enable/disable FFT over the dedicated FFT buffer.
        """
        if getattr(self, '_auto_fft_change', False):
            return
        self._fft_enabled = bool(checked)
        if self._fft_enabled:
            # Clear previous spectra so the next display corresponds only to
            # the current FFT-buffer selection.
            self._reset_fft_buffers()
            self.sigFftWorkerReset.emit(True, True)
            self._fft_chunk_window_idx = 0
            self._last_fft_compute_ts = 0.0
            self._fft_last_window_end_t = None
            self._last_fft_freq = None
            self._last_fft_mag_by_phys.clear()
            for curve in self._fft_curves_by_phys.values():
                try:
                    curve.clear()
                except Exception:
                    pass
            try:
                tgt = max(1.0, float(self._fft_duration_seconds))
                self.lblFftStatus.setText(f"FFT: finestra su disco 0.00/{tgt:g}s - F max= 0 Hz")
            except Exception:
                pass
            self._emit_fft_worker_config()
        else:
            self.sigFftWorkerReset.emit(True, True)
            self._clear_fft_plot_visuals(clear_cached=True)

    # ----------------------------- Definisci Tipo Risorsa -----------------------------
    def _open_resource_manager(self):
        try:
            from gestione_risorse import ResourceManagerDialog
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Errore", f"Impossibile importare gestione_risorse:\n{e}")
            return
        # Apri il dialog con il percorso corrente del database dei sensori
        try:
            dlg = ResourceManagerDialog(self.acq, xml_path=self._sensor_db_path, parent=self)
        except Exception:
            dlg = ResourceManagerDialog(self.acq, xml_path=SENSOR_DB_DEFAULT, parent=self)
        dlg.exec_()
        # Se l'utente ha cambiato il percorso del DB, aggiorna la variabile
        try:
            if dlg.xml_path:
                self._sensor_db_path = dlg.xml_path
        except Exception:
            pass
        # refresh liste e scale
        self._populate_type_column()
        self._recompute_all_calibrations()

    # ----------------------------- Workspace management -----------------------------
    def _workspace_supported_daq(self):
        try:
            value = str(getattr(self.acq, "board_type", "NI9234") or "NI9234").strip()
        except Exception:
            value = "NI9234"
        return value.upper() if value else "NI9234"

    def _workspace_current_device_name(self):
        try:
            return str((self.cmbDevice.currentData() or self.cmbDevice.currentText() or "")).strip()
        except Exception:
            return ""

    def _workspace_section_base(self, supported_daq, device_name):
        daq = str(supported_daq or "").strip().upper() or "NI9234"
        dev = str(device_name or "").strip()
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", dev).strip("_")
        if not normalized:
            normalized = "DEVICE"
        return f"ws.{daq}.{normalized}"

    def _workspace_find_entry_base(self, cfg, supported_daq, device_name):
        if not device_name:
            return None
        target_daq = str(supported_daq or "").strip().upper()
        target_dev = str(device_name or "").strip().lower()
        for sec in cfg.sections():
            if not sec.startswith("ws.") or not sec.endswith(".general"):
                continue
            gen = cfg[sec]
            sec_daq = str(gen.get("supporteddaq", "") or "").strip().upper()
            sec_dev = str(gen.get("device_name", "") or "").strip()
            if not sec_dev:
                continue
            if sec_daq == target_daq and sec_dev.lower() == target_dev:
                base = sec[: -len(".general")]
                if f"{base}.channels" in cfg:
                    return base
        return None

    def _workspace_next_free_base(self, cfg, base):
        candidate = base
        idx = 2
        while (f"{candidate}.general" in cfg) or (f"{candidate}.channels" in cfg):
            candidate = f"{base}_{idx}"
            idx += 1
        return candidate

    def _save_workspace(self):
        path = ""
        try:
            dlg = QtWidgets.QFileDialog(self, "Salva workspace")
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            dlg.setNameFilter("INI (*.ini)")
            dlg.setDefaultSuffix("ini")
            dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, True)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                files = dlg.selectedFiles() or []
                path = files[0] if files else ""
        except Exception:
            path = ""
        if not path:
            return
        if not path.lower().endswith(".ini"):
            path += ".ini"

        cfg = configparser.ConfigParser()
        if os.path.isfile(path):
            try:
                cfg.read(path, encoding="utf-8")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Errore", f"Impossibile leggere workspace:\n{e}")
                return

        supported_daq = self._workspace_supported_daq()
        device_name = self._workspace_current_device_name()
        if not device_name:
            QtWidgets.QMessageBox.critical(self, "Errore", "Nessun workspace per la scheda corrente")
            return

        entry_base = self._workspace_find_entry_base(cfg, supported_daq, device_name)
        if entry_base is not None:
            ans = QtWidgets.QMessageBox.question(
                self,
                "Conferma sovrascrittura",
                "Scheda già presente vuoi sovrascriverla?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
        else:
            proposed = self._workspace_section_base(supported_daq, device_name)
            entry_base = self._workspace_next_free_base(cfg, proposed)

        sec_general = f"{entry_base}.general"
        sec_channels = f"{entry_base}.channels"
        cfg[sec_general] = {}
        cfg[sec_channels] = {}

        gen = cfg[sec_general]
        try:
            gen["sensor_db_path"] = self._sensor_db_path or ""
        except Exception:
            gen["sensor_db_path"] = ""
        gen["supporteddaq"] = supported_daq
        gen["device_name"] = device_name
        gen["save_dir"] = self.txtSaveDir.text().strip()
        gen["base_filename"] = self.txtBaseName.text().strip()
        try:
            gen["ram_mb"] = str(int(self.spinRam.value()))
        except Exception:
            gen["ram_mb"] = ""
        gen["fs"] = (self.rateEdit.text() or "").strip()

        chsec = cfg[sec_channels]
        all_phys = []
        enabled_list = []
        type_list = []
        label_list = []
        zero_raw_list = []
        zero_display_list = []
        for r in range(self.table.rowCount()):
            try:
                phys = self.table.item(r, COL_PHYS).text()
            except Exception:
                phys = f"ai{r}"

            enable_item = self.table.item(r, COL_ENABLE)
            enabled = bool(enable_item and enable_item.checkState() == QtCore.Qt.Checked)

            cmb = self.table.cellWidget(r, COL_TYPE)
            typ = ""
            if isinstance(cmb, QtWidgets.QComboBox):
                try:
                    typ = cmb.currentText().strip()
                except Exception:
                    typ = ""

            lbl_item = self.table.item(r, COL_LABEL)
            label = lbl_item.text().strip() if lbl_item else ""

            try:
                baseline_raw = None
                if hasattr(self.acq, "_zero"):
                    baseline_raw = self.acq._zero.get(phys)
                if baseline_raw is None:
                    zero_raw_list.append("")
                else:
                    zero_raw_list.append(f"{float(baseline_raw):.12g}")
            except Exception:
                zero_raw_list.append("")

            try:
                zero_item = self.table.item(r, COL_ZERO_VAL)
                if zero_item:
                    zero_display_list.append(zero_item.text().strip())
                else:
                    zero_display_list.append("")
            except Exception:
                zero_display_list.append("")

            all_phys.append(phys)
            enabled_list.append("1" if enabled else "0")
            type_list.append(typ)
            label_list.append(label)

        chsec["phys"] = ",".join(all_phys)
        chsec["enabled"] = ",".join(enabled_list)
        chsec["type"] = ",".join(type_list)
        chsec["label"] = ",".join(label_list)
        chsec["zero_raw"] = ",".join(zero_raw_list)
        chsec["zero_display"] = ",".join(zero_display_list)

        try:
            with open(path, "w", encoding="utf-8") as f:
                cfg.write(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Errore", f"Impossibile salvare il workspace:\n{e}")
            return
        QtWidgets.QMessageBox.information(self, "Salvato", f"Workspace salvato:\n{path}")

    def _load_workspace(self):
        try:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Carica workspace", "", "INI (*.ini)"
            )
        except Exception:
            fname = ""
        if not fname:
            return

        cfg = configparser.ConfigParser()
        try:
            cfg.read(fname, encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Errore", f"Impossibile leggere workspace:\n{e}")
            return

        supported_daq = self._workspace_supported_daq()
        current_device = self._workspace_current_device_name()
        if not current_device:
            QtWidgets.QMessageBox.warning(self, "Attenzione", "Nessun workspace per la scheda corrente")
            return

        entry_base = self._workspace_find_entry_base(cfg, supported_daq, current_device)
        if entry_base is None:
            QtWidgets.QMessageBox.warning(self, "Attenzione", "Nessun workspace per la scheda corrente")
            return

        sec_general = f"{entry_base}.general"
        sec_channels = f"{entry_base}.channels"
        if sec_general not in cfg or sec_channels not in cfg:
            QtWidgets.QMessageBox.warning(self, "Attenzione", "Nessun workspace per la scheda corrente")
            return

        gen = cfg[sec_general]
        chsec = cfg[sec_channels]

        sensor_db = gen.get("sensor_db_path", "").strip()
        if sensor_db:
            self._sensor_db_path = sensor_db

        sd = gen.get("save_dir", "").strip()
        if sd:
            self._save_dir = sd
            self.txtSaveDir.setText(sd)

        bn = gen.get("base_filename", "").strip()
        if bn:
            self._base_filename = bn
            self.txtBaseName.setText(bn)

        try:
            ram_mb = int(gen.get("ram_mb", "").strip())
            if ram_mb > 0:
                self.spinRam.setValue(ram_mb)
        except Exception:
            pass

        fs_txt = gen.get("fs", "").strip()
        if fs_txt:
            self.rateEdit.setText(fs_txt)
            try:
                self._on_rate_edit_finished()
            except Exception:
                pass
        self._sync_emit_status_update()

        self._populate_type_column()
        self._recompute_all_calibrations()

        phys_raw = chsec.get("phys", "")
        enabled_raw = chsec.get("enabled", "")
        type_raw = chsec.get("type", "")
        label_raw = chsec.get("label", "")
        phys_list = [p.strip() for p in phys_raw.split(",")] if phys_raw else []
        enabled_list = [e.strip() for e in enabled_raw.split(",")] if enabled_raw else []
        type_list = [t.strip() for t in type_raw.split(",")] if type_raw else []
        label_list = [l.strip() for l in label_raw.split(",")] if label_raw else []
        zero_raw_raw = chsec.get("zero_raw", "")
        zero_display_raw = chsec.get("zero_display", "")
        zero_raw_list = [z.strip() for z in zero_raw_raw.split(",")] if zero_raw_raw else []
        zero_display_list = [z.strip() for z in zero_display_raw.split(",")] if zero_display_raw else []

        baseline_raw_map = {}
        baseline_disp_map = {}
        self._populate_table()
        self._populate_type_column()

        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            phys = None
            try:
                phys = self.table.item(r, COL_PHYS).text()
            except Exception:
                phys = None
            if not phys:
                continue

            try:
                idx = phys_list.index(phys)
            except Exception:
                idx = -1

            enable_item = self.table.item(r, COL_ENABLE)
            if enable_item:
                state = False
                if idx >= 0 and idx < len(enabled_list):
                    state = enabled_list[idx].strip().lower() in ("1", "true")
                enable_item.setCheckState(QtCore.Qt.Checked if state else QtCore.Qt.Unchecked)

            cmb = self.table.cellWidget(r, COL_TYPE)
            if isinstance(cmb, QtWidgets.QComboBox):
                if idx >= 0 and idx < len(type_list):
                    tval = type_list[idx] or "Voltage"
                    pos = cmb.findText(tval)
                    if pos >= 0:
                        cmb.setCurrentIndex(pos)
                    else:
                        cmb.setCurrentIndex(0)
                else:
                    cmb.setCurrentIndex(0)

            it = self.table.item(r, COL_LABEL)
            if it is None:
                it = QtWidgets.QTableWidgetItem("")
                self.table.setItem(r, COL_LABEL, it)
            if idx >= 0 and idx < len(label_list):
                it.setText(label_list[idx])
            else:
                it.setText(phys)

            if idx >= 0:
                if idx < len(zero_raw_list):
                    baseline_raw_map[phys] = zero_raw_list[idx]
                if idx < len(zero_display_list):
                    baseline_disp_map[phys] = zero_display_list[idx]
        self.table.blockSignals(False)

        self._recompute_all_calibrations()
        self._update_acquisition_from_table()
        try:
            for r in range(self.table.rowCount()):
                phys_item = self.table.item(r, COL_PHYS)
                if phys_item is None:
                    continue
                phys = phys_item.text()
                br = baseline_raw_map.get(phys, "")
                baseline_value = None
                if br not in ("", None):
                    try:
                        baseline_value = float(br)
                    except Exception:
                        baseline_value = None
                try:
                    self.acq.set_zero_raw(phys, baseline_value)
                except Exception:
                    pass
                zero_disp = baseline_disp_map.get(phys, "")
                if zero_disp:
                    self.table.item(r, COL_ZERO_VAL).setText(zero_disp)
                else:
                    try:
                        self.table.item(r, COL_ZERO_VAL).setText("0.0")
                    except Exception:
                        pass
        except Exception:
            pass
        QtWidgets.QMessageBox.information(self, "Caricato", f"Workspace caricato:\n{fname}")

    # ----------------------------- Channel helpers per ResourceManager -----------------------------
    def is_channel_enabled(self, phys: str) -> bool:
        """Restituisce True se il canale fisico ? abilitato nella tabella."""
        try:
            for r in range(self.table.rowCount()):
                phys_item = self.table.item(r, COL_PHYS)
                if phys_item and phys_item.text() == phys:
                    # colonna abilita ? una QTableWidgetItem con stato di check
                    enable_item = self.table.item(r, COL_ENABLE)
                    if enable_item:
                        return enable_item.checkState() == QtCore.Qt.Checked
                    return False
        except Exception:
            pass
        return False

    def enable_physical_channel(self, phys: str):
        """Abilita il canale fisico nella tabella."""
        try:
            for r in range(self.table.rowCount()):
                phys_item = self.table.item(r, COL_PHYS)
                if phys_item and phys_item.text() == phys:
                    enable_item = self.table.item(r, COL_ENABLE)
                    if enable_item and enable_item.checkState() != QtCore.Qt.Checked:
                        enable_item.setCheckState(QtCore.Qt.Checked)
                    break
        except Exception:
            pass

    # ----------------------------- Valore istantaneo in tabella -----------------------------
    def _update_table_value(self, start_label_name, val_volt_zeroed):
        # mappa dal nome al phys usato allo start
        phys = None
        for p, nm in self._start_label_by_phys.items():
            if nm == start_label_name:
                phys = p; break
        if phys is None:
            return
        r = self._find_row_by_phys(phys)
        if r < 0:
            return
        cal = self._calib_by_phys.get(phys, {"a":1.0,"b":0.0,"unit":""})
        a = float(cal.get("a",1.0)); b = float(cal.get("b",0.0))
        unit = cal.get("unit","")
        eng = a * float(val_volt_zeroed) + b
        text = f"{eng:.6g}" + (f" {unit}" if unit else "")
        self._auto_change = True
        self.table.item(r, COL_VALUE).setText(text)
        self._auto_change = False

    # ----------------------------- Chiusura ordinata -----------------------------
    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            if self._sync_agent is not None:
                self._sync_agent.close()
                self._sync_agent.deleteLater()
                self._sync_agent = None
        except BaseException:
            pass
        try:
            self._save_config()
        except BaseException:
            pass
        # stop timer UI
        try:
            if self.guiTimer.isActive():
                self.guiTimer.stop()
        except BaseException:
            pass
        try:
            if hasattr(self, "_count_timer") and self._count_timer.isActive():
                self._count_timer.stop()
        except BaseException:
            pass
        try:
            if hasattr(self, "_backlog_timer") and self._backlog_timer.isActive():
                self._backlog_timer.stop()
        except BaseException:
            pass
        # ferma core
        try:
            self.acq.set_recording(False)
        except BaseException:
            pass
        try:
            self.acq.stop()
        except BaseException:
            pass
        try:
            self.sigFftWorkerReset.emit(True, True)
        except BaseException:
            pass
        try:
            self.sigFftWorkerBlock.disconnect()
        except BaseException:
            pass
        try:
            self.sigFftWorkerConfig.disconnect()
        except BaseException:
            pass
        try:
            self.sigFftWorkerReset.disconnect()
        except BaseException:
            pass
        try:
            if hasattr(self, "_fft_worker") and self._fft_worker is not None:
                try:
                    QtCore.QMetaObject.invokeMethod(
                        self._fft_worker,
                        "deleteLater",
                        QtCore.Qt.QueuedConnection,
                    )
                except Exception:
                    pass
            if hasattr(self, "_fft_thread") and self._fft_thread is not None:
                try:
                    self._fft_thread.requestInterruption()
                except Exception:
                    pass
                self._fft_thread.quit()
                deadline = time.monotonic() + 8.0
                while self._fft_thread.isRunning() and time.monotonic() < deadline:
                    self._fft_thread.wait(100)
                    QtWidgets.QApplication.processEvents()
                self._fft_thread.deleteLater()
                self._fft_thread = None
            self._fft_worker = None
        except BaseException:
            pass
        # disconnetti segnali
        try:
            self.sigInstantBlock.disconnect()
        except BaseException:
            pass
        try:
            self.sigChartPoints.disconnect()
        except BaseException:
            pass
        try:
            self.channelValueUpdated.disconnect()
        except BaseException:
            pass
        super().closeEvent(event)

    def _on_zero_button_clicked(self, phys: str):
        """
        Azzeramento canale:
        - Legge il valore istantaneo ATTUALE (in unit? ingegneristiche)
        - Lo mostra in colonna 'Valore azzerato'
        - Fissa lo zero nel core come valore RAW (Volt) dell'istante
        """
        # riga del canale
        r = self._find_row_by_phys(phys)
        if r < 0:
            return

        # 1) valore istantaneo in unit? ingegneristiche (quello che vedi in UI)
        try:
            val_eng = self.acq.get_last_engineered(phys)
        except Exception:
            val_eng = None

        # unit? per visualizzazione
        unit = self._calib_by_phys.get(phys, {}).get("unit", "")
        if val_eng is not None:
            txt = f"{val_eng:.6g}" + (f" {unit}" if unit else "")
            self._auto_change = True
            self.table.item(r, COL_ZERO_VAL).setText(txt)
            self._auto_change = False

        # 2) imposta lo zero nel core come baseline RAW (Volt)
        try:
            last_raw = self.acq.get_last_raw(phys)
            if last_raw is not None:
                self.acq.set_zero_raw(phys, last_raw)
        except Exception:
            pass

    def _set_row_bg(self, row: int, col: int, color: QtGui.QColor):
        item = self.table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem("")
            self.table.setItem(row, col, item)
        item.setBackground(color)

    def _set_table_lock(self, lock: bool):
        """
        Blocca/sblocca le 5 colonne: Abilita, Canale fisico, Tipo risorsa,
        Nome canale, Valore istantaneo. Grigio chiaro quando lock=True.
        """
        gray = QtGui.QColor("#e9ecef")
        white = QtGui.QColor("#ffffff")
        nrows = self.table.rowCount()

        for r in range(nrows):
            # --- Abilita (QTableWidgetItem con checkbox) ---
            it = self.table.item(r, COL_ENABLE)
            if it:
                if lock:
                    # rimuovo la possibilit? di spuntare
                    it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                else:
                    # riabilito la spunta
                    it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            self._set_row_bg(r, COL_ENABLE, gray if lock else white)

            # --- Canale fisico (sempre non editabile; solo colore) ---
            it = self.table.item(r, COL_PHYS)
            if it:
                it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self._set_row_bg(r, COL_PHYS, gray if lock else white)

            # --- Tipo risorsa (QComboBox) ---
            w = self.table.cellWidget(r, COL_TYPE)
            if isinstance(w, QtWidgets.QComboBox):
                w.setEnabled(not lock)
            self._set_row_bg(r, COL_TYPE, gray if lock else white)

            # --- Nome canale (item editabile quando unlock) ---
            it = self.table.item(r, COL_LABEL)
            if it:
                if lock:
                    it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                else:
                    it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
            self._set_row_bg(r, COL_LABEL, gray if lock else white)

            # --- Valore istantaneo (display only; solo colore) ---
            self._set_row_bg(r, COL_VALUE, gray if lock else white)

    def _uncheck_all_enabled(self):
        """Rimuove tutte le spunte 'Abilita' (senza scatenare ricalcoli ripetuti)."""
        self._auto_change = True
        try:
            nrows = self.table.rowCount()
            for r in range(nrows):
                it = self.table.item(r, COL_ENABLE)
                if it and it.flags() & QtCore.Qt.ItemIsUserCheckable:
                    it.setCheckState(QtCore.Qt.Unchecked)
        finally:
            self._auto_change = False
        # applica lo stato all'acquisizione
        self._update_acquisition_from_table()



