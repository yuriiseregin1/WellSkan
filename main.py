# Имопрт библиотек

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import seaborn as sn;
sn.set(font_scale=1.4)
import cv2
from torchvision import datasets, models, transforms
import torch, torch.nn as nn
from PIL import Image
import sys
from PyQt5.QtWidgets import (QFileDialog, QApplication)
import os
import numpy as np


# Основной класс

class Main(QWidget):
    def __init__(self):
        super(Main, self).__init__()
        self.setWindowTitle('WellSkan')
        self.setFixedSize(640, 490)
        self.setWindowIcon(QtGui.QIcon('background.png'))
        pal = self.palette()
        pal.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                     QtGui.QBrush(QtGui.QPixmap("background.png")))
        self.setPalette(pal)
        self.get_link_text = QLabel('WellSkan', self)
        self.get_link_text.move(0, 10)
        self.get_link_text.setFixedSize(640, 20)
        self.get_link_text.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        self.get_link_text.setAlignment(Qt.AlignCenter)
        self.compare1 = QPushButton(self)
        self.compare1.resize(580, 190)
        self.compare1.move(30, 45)
        self.compare1.setText('Определить слой с помощью нейросети')
        self.compare1.setStyleSheet("background-color: rgba(180, 180, 180, 100)")
        self.compare1.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.compare1.clicked.connect(self.open_process)
        self.compare2 = QPushButton(self)
        self.compare2.resize(275, 190)
        self.compare2.move(30, 265)
        self.compare2.setText('Как это работает')
        self.compare2.setStyleSheet("background-color: rgba(180, 180, 180, 100)")
        self.compare2.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.compare2.clicked.connect(self.open_HowItWork)

        self.compare3 = QPushButton(self)
        self.compare3.resize(275, 190)
        self.compare3.move(335, 265)
        self.compare3.setText('О нас')
        self.compare3.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.compare3.clicked.connect(self.open_about)
        self.compare3.setStyleSheet("background-color: rgba(180, 180, 180, 100)")

    def open_process(self):
            self.wnd = CreateRequest()
            self.close()
            self.wnd.show()

    def open_about(self):
        try:
            self.wnd = About()
            self.close()
            self.wnd.show()
        except Exception as E:
            print("Произошла ошибка. Попробуйте еще раз или обратитесь к разработчику. -", E)

    def open_HowItWork(self):
        try:
            self.wnd = HowItWork()
            self.close()
            self.wnd.show()
        except Exception as E:
            print("Произошла ошибка. Попробуйте еще раз или обратитесь к разработчику. -", E)


class CreateRequest(QWidget):
    def __init__(self):
        super(CreateRequest, self).__init__()
        self.setWindowTitle('WellSkan')
        self.setFixedSize(640, 490)
        self.setWindowIcon(QtGui.QIcon('background.png'))
        pal = self.palette()
        pal.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                     QtGui.QBrush(QtGui.QPixmap("background.png")))
        self.setPalette(pal)

        self.compare = QPushButton(self)
        self.compare.resize(80, 20)
        self.compare.move(15, 450)
        self.compare.setText('<- Назад')
        self.compare.setStyleSheet("background-color: rgba(180, 180, 180, 100)")
        self.compare.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.compare.clicked.connect(self.close_window)

        self.showDialog()

        with open('result.txt', 'r') as f:
            result = f.read()

        self.output = QLabel('Результат обработки изображения - ' + str(result), self)
        self.output.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        self.output.setStyleSheet("background-color: rgba(180, 180, 180, 160)")
        layout = QVBoxLayout(self)

        layout.addWidget(self.output)

        layout.addWidget(self.compare)
        self.output.setAlignment(Qt.AlignCenter)

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        f = open(fname, 'r')
        print(f.name)
        result = (Request.process(self, str(f.name)))
        print(result)
        with open('result.txt', 'w') as f:
            f.write(result)

    def close_window(self):
        self.wnd = Main()
        self.close()
        self.wnd.show()


class OpenResult(QWidget):
    def __init__(self):
        super(OpenResult, self).__init__()
        self.setWindowTitle('WellSkan')
        print(22)
        self.setFixedSize(640, 490)
        self.setWindowIcon(QtGui.QIcon('background.png'))
        pal = self.palette()
        pal.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                     QtGui.QBrush(QtGui.QPixmap("background.png")))
        self.setPalette(pal)


class HowItWork(QWidget):
    def __init__(self):
        super(HowItWork, self).__init__()
        self.setWindowTitle('WellSkan')
        self.setFixedSize(640, 490)
        self.setWindowIcon(QtGui.QIcon('background.png'))
        pal = self.palette()
        pal.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                     QtGui.QBrush(QtGui.QPixmap("background.png")))
        self.setPalette(pal)

        self.compare = QPushButton(self)
        self.compare.resize(80, 20)
        self.compare.move(15, 450)
        self.compare.setText('<- Назад')
        self.compare.setStyleSheet("background-color: rgba(180, 180, 180, 100)")
        self.compare.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.compare.clicked.connect(self.close_window)

        self.output = QLabel('По средствам использование информационных технологий,\n' +
                             'а именно нейронных сетей на базе предобученной модели vgg19...',
                             self)
        self.output.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        self.output.setStyleSheet("background-color: rgba(180, 180, 180, 160)")
        layout = QVBoxLayout(self)
        layout.addWidget(self.output)
        layout.addWidget(self.compare)
        self.output.setAlignment(Qt.AlignCenter)

    def close_window(self):
        self.wnd = Main()
        self.close()
        self.wnd.show()


class About(QWidget):
    def __init__(self):
        super(About, self).__init__()
        self.setWindowTitle('WellSkan')
        self.setFixedSize(640, 490)
        self.setWindowIcon(QtGui.QIcon('background.png'))
        pal = self.palette()
        pal.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                     QtGui.QBrush(QtGui.QPixmap("background.png")))
        self.setPalette(pal)

        self.compare = QPushButton(self)
        self.compare.resize(80, 20)
        self.compare.move(15, 450)
        self.compare.setText('<- Назад')
        self.compare.setStyleSheet("background-color: rgba(180, 180, 180, 100)")
        self.compare.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.compare.clicked.connect(self.close_window)

        self.output = QLabel('Здравствуйте, мы команда из Санкт-Петербурга,\n' +
                            'в состав которой входят\n' +
                            'Ильина Александра,\n' +
                            'Серёгин Юрий и\n' +
                            'Бурдуленко Ольга.\n' +
                            'В настоящее время появляется всё больше\n' +
                            'нефтегазовых месторождений со сложным\n' +
                            'геологическим строением. Чтобы\n' +
                            'создать корректную модель\n' +
                            'месторождения, а затем подробно\n' +
                            'её изучить, геологам требуется много\n' +
                            'сил и времени. По поручению\n' +
                            'Газпромнефть НТЦ мы создали\n' +
                            'приложение, которое позволит\n' +
                            'петрофизикам производить\n' +
                            'наиболее быстрый и точный\n' +
                            'анализ данных, полученных\n' +
                            'со скважинных микросканеров.\n', self)

        self.output.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        self.output.setStyleSheet("background-color: rgba(180, 180, 180, 160)")
        layout = QVBoxLayout(self)
        layout.addWidget(self.output)
        layout.addWidget(self.compare)
        self.output.setAlignment(Qt.AlignCenter)

    def close_window(self):
        self.wnd = Main()
        self.close()
        self.wnd.show()


# класс обработки изображения

class Request(QWidget):
    def __init__(self):
        super(Request, self).__init__()
        self.setWindowTitle('WellSkan')
        self.setFixedSize(600, 450)
        self.setWindowIcon(QtGui.QIcon('background.png'))
        pal = self.palette()
        pal.setBrush(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                     QtGui.QBrush(QtGui.QPixmap("background.png")))
        self.setPalette(pal)
        with open('filename.txt', 'r') as f:
            file = f.read()
        result = (self.process(file))
        self.res = QLabel(f'Результат: {result}', self)
        self.res.move(230, 150)
        self.res.setStyleSheet("background-color: rgba(180, 180, 180, 160)")

    def process(self, file):
        # чтение из таблицы
        def get_result(path):
            print("Обработка...")
            print("Завершено 0%")
            decode = {0: 'Вывалы',
                      1: 'Устественные_трещины',
                      2: 'Каверны',
                      3: 'Ракушняк_карбонаты',
                      4: 'Слои',
                      5: 'Техногенные_трещины'
                      }
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            print("Завершено 10%")
            print(path)
            image = cv2.imread(path)
            print("Завершено 20%")
            image = cv2.resize(image, (700, 800))  # менять здесь
            print("Завершено 30%")
            image = abs(image - 255)
            print("Завершено 40%")
            image = Image.fromarray(image)
            print("Завершено 50%")
            image = transform(image)
            print("Завершено 60%")
            image = image.unsqueeze(0)
            print("Завершено 70%")
            logits = model(image)
            print("Завершено 80%")
            label = int(logits.max(1)[1].data.numpy())
            print("Завершено 90%")
            print("Завершено 100%")
            return decode[label]
        try:
            model = models.vgg19_bn(pretrained=True)
            model.classifier[6] = nn.Linear(in_features=4096, out_features=6, bias=True)
            model.eval()
            model.load_state_dict(torch.load('vgg_19_bn_adam_1e-4_700x800.zip',
                                             map_location=torch.device('cpu')))
            return (get_result(file))
        except:
            self.Main().show()
            return 0



with open('result.txt', 'w') as f:
    f.write(str('Ошибка'))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    wnd = Main()
    wnd.show()
    sys.exit(app.exec())

