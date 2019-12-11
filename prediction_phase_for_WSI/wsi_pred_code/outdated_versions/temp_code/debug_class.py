class model():
    def __init__(self):
        self.pair_list=[]

    def add_list(self):
        for i in range(10):
            self.pair_list.append(i)

M=model()
M.add_list()
print(M.pair_list)
M.add_list()
print(M.pair_list)
