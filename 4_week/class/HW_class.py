import random

class MyDataLoader2:
    def __init__(self, data1, data2):
        # data1과 data2 크기는 같다고 가정합시다
        self.size = len(data1)
        self.data1 = data1
        self.data2 = data2
        ###### BLANK ####

    # __iter__에서는 나를 반환해주고 __next__에서는 for loop가
    # 돌떄마다 인쇄하고 싶은 값을 리턴합니다.
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.size:
            r = self.data1[self.index] + self.data2[self.index]
            self.index += 1
            return r
        else:
            raise StopIteration

    def shuffle(self):
        random.shuffle(self.data1)
        random.shuffle(self.data2)

    def make_square(self):
        self.data1 = [i * i for i in self.data1]
        self.data2 = [i * i for i in self.data2]



# data1과 data2 크기는 같다고 가정합시다
loader = MyDataLoader2([i for i in range(300)], [i for i in range(300)])
loader.make_square()  ## data1과 data2를 제곱할 수 있는 method를 완성해주세요!
loader.shuffle()
for i, x in enumerate(loader):  # for loop 가 도니까 __iter__
    print("{} : {}".format(i, x))  # i번째 data1과 data2를 더한 값을 리턴하도록 만들어주세요!
