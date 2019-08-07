from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QLineEdit, QTextEdit, QPushButton
from PyQt5.QtWidgets import QGridLayout, QDesktopWidget
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon

import os
import sys
import pandas as pd

from mainwindow import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())