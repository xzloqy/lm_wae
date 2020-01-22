# dict = {'a': 1}

# dict['a'] += 1
# dict['b'] = 1
# for i in range(10):
#     dict['b'] += 1

# print(dict)


class test():
    def __init__(self):
        self.myloss = {}

    def train(self):
        self.myloss['lossa'] = 0
        self.myloss['lossb'] = 0

        for i in range(10):
            self.myloss['lossa'] += 1.2
            self.myloss['lossb'] += 1

    def run(self):
        self.train()
        print("loss: {} ".format(self.myloss))
        # return self.myloss


if __name__ == '__main__':
    testa = test()
    print("begin run")
    testa.run()
    # print(myloss)
