from PyQt5.QtCore import QThread, pyqtSignal

class updateList(QThread):

    def __init__(self, ids, df):
        super().__init__(self)
        self.ids = ids
        self.df = df
        countCharged = pyqtSignal(int)

    def run(self):
        count = 0
        tmp = []
        for i in range(self.ids):
            id = self.ids[i]
            if '.' in id:
                id = id.split('.')
                dx = self.df[(self.df['RID'] == int(id[0])) & (self.df['MRI ImageID'] == int(id[1]))]['DX'].values[0]
            else:
                dx = self.df[(self.df['RID'] == int(id)) & (self.df['MRI ImageID'] == "")]['DX'].values[0]
            # train on AD/MCI/NL ([1,2,3]) or only AD/NL ([1,3])
            if dx in [1, 3]: tmp.append(self.ids_train[i])
            self.countCharged.emit(i)
        return