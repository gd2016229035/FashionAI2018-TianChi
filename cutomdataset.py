from mxnet import gluon, image, init, nd
import random
import os
class custom_dataset(gluon.data.Dataset):
    def unique_index(self, L, e):  # return the index of e
        index = []
        for (i, j) in enumerate(L):
            if j == e:
                index.append(i)
        return index

    def __init__(self, root, filename, transform=None):
        self.image_path_list = []
        self.label_list = []
        self._transform = transform
        self.items = []
        with open(filename, 'r') as f:
            train = f.readlines()
            random.shuffle(train)
            for line in train:
                tmp = line.strip().split(',')
                image_file_name = os.path.join(root, tmp[0])
                label = tmp[1]
                label_length = len(label)
                y = list(label).index('y')
                m = self.unique_index(list(label), 'm')
                label_final = [0] * label_length
                if len(m) == 0:
                    label_final[y] = 1
                elif len(m) == 1:
                    label_final[y] = 0.8
                    label_final[m[0]] = 0.2
                elif len(m) == 2:
                    label_final[y] = 0.7
                    label_final[m[0]] = 0.15
                    label_final[m[1]] = 0.15
                elif len(m) == 3:
                    label_final[y] = 0.7
                    label_final[m[0]] = 0.1
                    label_final[m[1]] = 0.1
                    label_final[m[2]] = 0.1
                else:
                    label_final[y] = 0.6
                    pro = 0.4 / len(m)
                    for i in m:
                        label_final[i] = pro
                self.items.append((image_file_name, label_final))
        #print(self.items)

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0])
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)

#muti-hot-softmax
class custom_dataset2(gluon.data.Dataset):

    def unique_index(self, L, e):  # return the index of e
        index = []
        for (i, j) in enumerate(L):
            if j == e:
                index.append(i)
        return index

    def __init__(self, root, filename, transform=None):
        self.image_path_list = []
        self.label_list = []
        self._transform = transform
        self.items = []
        with open(filename, 'r') as f:
            train = f.readlines()
            #random.shuffle(train)
            for line in train:
                tmp = line.strip().split(',')
                image_file_name = os.path.join(root, tmp[0])
                label = tmp[1]
                label_length = len(label)
                y = list(label).index('y')
                m = self.unique_index(list(label), 'm')
                label_final = [99] * label_length
                label_final[0] = y
                n = 1
                for i in m:
                    label_final[n] = i
                    n += 1
                self.items.append((image_file_name, label_final))
        #print(self.items)

    def __getitem__(self, idx):
        #print(self.items[idx][0])
        img = image.imread(self.items[idx][0])
        label = self.items[idx][1]
        # if len(label)!=1:
        #     print('mmmmmmm')
        #abel_m = self.items[idx][2]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)