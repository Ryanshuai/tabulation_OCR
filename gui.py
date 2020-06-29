# encoding:utf8

import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.widget = ImageWithMouseControl(self)
        self.setWindowTitle('Image with mouse control')
        self.move(0, 0)


class ImageWithMouseControl(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.img = QPixmap('./ren.png')
        self.scaled_img = self.img.scaled(self.size())
        self.point = QPoint(0, 0)
        self.setGeometry(0, 0, self.img.width(), self.img.height())

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        self.draw_img(painter)
        painter.end()

    def draw_img(self, painter):
        painter.drawPixmap(self.point, self.scaled_img)

    def mouseMoveEvent(self, e):
        if self.right_hold:
            self._endPos = e.pos() - self._startPos
            self.point = self.point + self._endPos
            self._startPos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.right_hold = True
            self._startPos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self.right_hold = False
        elif e.button() == Qt.LeftButton:
            self.point = QPoint(0, 0)
            self.scaled_img = self.img.scaled(self.size())
            self.repaint()

    def wheelEvent(self, e):
        if e.angleDelta().y() < 0:
            self.scaled_img = self.img.scaled(self.scaled_img.width() * 0.9, self.scaled_img.height() * 0.9)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() / 0.9)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() / 0.9)
            self.point = QPoint(new_w, new_h)
            self.repaint()
        elif e.angleDelta().y() > 0:
            self.scaled_img = self.img.scaled(self.scaled_img.width() / 0.9, self.scaled_img.height() / 0.9)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() * 0.9)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() * 0.9)
            self.point = QPoint(new_w, new_h)
            self.repaint()

    def resizeEvent(self, e):
        if self.parent is not None:
            self.scaled_img = self.img.scaled(self.size())
            self.point = QPoint(0, 0)
            self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()
    # ex = ImageWithMouseControl()
    ex.show()
    app.exec_()
