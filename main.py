import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from ui import AcquisitionWindow
from acquisition import AcquisitionManager
from tdms_merge import TdmsMerger

def main():
    app = QtWidgets.QApplication(sys.argv)
    # Ask the user which board model they wish to use. This dialog appears
    # only at startup and determines whether the application will target an
    # NI‑9201 (8 channels) or an NI‑9234 (4 channels). The selection is used
    # to instantiate the AcquisitionManager with the appropriate configuration.
    class BoardSelectionDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Seleziona modello scheda DAQ")
            self.resize(380, 160)
            layout = QtWidgets.QVBoxLayout(self)
            self.label = QtWidgets.QLabel(
                "Seleziona il modello di scheda DAQ da utilizzare.\n"
                "• NI‑9201: 8 canali analogici, ±10 V, 12 bit, 500 kS/s aggregati.\n"
                "• NI‑9234: 4 canali simultanei, ±5 V, 24 bit, fino a 51,2 kS/s per canale.\n\n"
                "La scelta influisce sul numero di canali visualizzati e sui limiti di frequenza."
            )
            self.label.setWordWrap(True)
            layout.addWidget(self.label)
            self.grp = QtWidgets.QButtonGroup(self)
            rb9201 = QtWidgets.QRadioButton("NI‑9201")
            rb9234 = QtWidgets.QRadioButton("NI‑9234")
            rb9234.setChecked(True)
            self.grp.addButton(rb9201)
            self.grp.addButton(rb9234)
            h = QtWidgets.QHBoxLayout()
            h.addWidget(rb9201)
            h.addWidget(rb9234)
            layout.addLayout(h)
            btnBox = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            layout.addWidget(btnBox)
            btnBox.accepted.connect(self.accept)
            btnBox.rejected.connect(self.reject)
            self._selected = "NI9234"
            rb9201.toggled.connect(lambda checked: self._set_sel("NI9201") if checked else None)
            rb9234.toggled.connect(lambda checked: self._set_sel("NI9234") if checked else None)
        def _set_sel(self, sel):
            self._selected = sel
        def selected(self):
            return self._selected

    dlg = BoardSelectionDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return
    board = dlg.selected()
    acq_manager = AcquisitionManager(board_type=board)
    window = AcquisitionWindow(acq_manager=acq_manager, merger=TdmsMerger())
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
