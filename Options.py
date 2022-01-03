class option(object):
    def __init__(self):
        self.epoch = 50
        self.lr = 1e-3
        self.bathSZ = 1
        self.root = 'D:/sideproject/camvid/dataset'
        self.outPath = ''
        self.H = 256
        self.W = 256
        self.train_cnt = 367
        self.test_cnt = 0
        self.num_classes = 12