#miaoooooo

import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from ui import AcquisitionWindow
from acquisition import AcquisitionManager
from tdms_merge import TdmsMerger

import os, sys
print("CWD:", os.getcwd())
print("MAIN:", __file__)
print("PYTHON:", sys.executable)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # In questa versione l’applicazione è progettata esclusivamente per la NI‑9234.
    # Non viene richiesto all’utente di selezionare il modello della scheda.
    acq_manager = AcquisitionManager()
    window = AcquisitionWindow(acq_manager=acq_manager, merger=TdmsMerger())
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
