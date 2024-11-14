# Create a QMessageBox instance

from pyqtgraph.Qt.QtWidgets import QMessageBox
import pyqtgraph as pg


def MessageBox(
    message,
    title="PyAbr3 Error",
    icon=QMessageBox.Icon.Information,
    buttons=QMessageBox.StandardButton.Ok,
    defaultButton=None,
):
    msgBox = QMessageBox()
    msgBox.setIcon(icon)
    msgBox.setText(title + "\n" + message)
    msgBox.setWindowTitle(title)  # ignored on mac OS so we add it to the text
    msgBox.setStandardButtons(buttons)
    msgBox.setDefaultButton(defaultButton)
    msgBox.setStyleSheet("QLabel{min-width: 200px; text-color: Qt.QColor.yellow}; Background-color: Qt.QColor.yellow")
    msgBox.show()
    # Show the message box and get the result
    result = msgBox.exec()

    # Handle the result
    if result == QMessageBox.StandardButton.Ok:
        return
    elif result == QMessageBox.StandardButton.Cancel:
        return
