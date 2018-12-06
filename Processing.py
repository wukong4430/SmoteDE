import numpy as np
import pandas as pd
from sklearn.utils import resample
import os


class Processing():

    def __init__(self):
        self.folder_name = "data"

    def import_data(self):
        '''

        读取文件夹中所有文件数据

        folder_name:文件夹的名字

        return: 文件夹下所有文件的数据

        '''

        dataset = pd.core.frame.DataFrame()

        # In Mac the path use '/' to identify the secondary path
        folder_path = self.folder_name + '//'

        for root, dirs, files in os.walk(folder_path):

            for file in files:
                file_path = os.path.join(root, file)

                data1 = pd.read_csv(file_path)

                dataset = dataset.append(data1, ignore_index=True)

        return dataset

    def import_single_data(self):
        '''
        单独读取文件夹中的每一个数据集
        folder_name: 文件夹的名字
        return: 文件夹下单独数据集的数据
        '''

        dataset = pd.core.frame.DataFrame()

        folder_path = self.folder_name + '/'

        for root, dirs, files in os.walk(folder_path):

            for file in files:
                file_path = os.path.join(root, file)

                dataset = pd.read_csv(file_path)

                yield dataset, file
    
    def import_single_data_withduo(self):
        """
        成对的读取数据，
        数据存放结构：
        /dataset/1/camel-1.0.csv
        /dataset/1/camel-1.2.csv
        /dataset/2/camel-1.2.csv
        /dataset/2/camel-1.4.csv
        ...

        return: trainx, trainy, testx, testy, foldername, trainname, testname
        """
        dataset_train = pd.core.frame.DataFrame()
        dataset_test = pd.core.frame.DataFrame()

        folder_path = self.folder_name + '/'

        def transform_data(original_data):
            original_data = original_data.iloc[:, 3:]

            original_data = np.array(original_data)

            k = len(original_data[0])

            # 降序排列train_data
            original_data = sorted(
                original_data, key=lambda x: x[-1], reverse=True)

            original_data = np.array(original_data)
            print(original_data.shape)
            original_data_X = original_data[:, 0:k - 1]

            original_data_y = original_data[:, k - 1]

            return original_data_X, original_data_y

        for root, dirs, files, in os.walk(folder_path):

            if root == 'dataset/':
                print(dirs)
                thisroot = root
                for dir in dirs:
                    dir_path = os.path.join(thisroot, dir)

                    for root, dirs, files, in os.walk(dir_path):
                        file_path_train = os.path.join(dir_path, files[0])
                        file_path_test = os.path.join(dir_path, files[1])
                        # print(file_path_test)
                        dataset_train = pd.read_csv(file_path_train)
                        dataset_test = pd.read_csv(file_path_test)

                        training_data_x, training_data_y = transform_data(
                            dataset_train)
                        testing_data_x, testing_data_y = transform_data(
                            dataset_test)
                        yield training_data_x, training_data_y, testing_data_x, testing_data_y, dir, files[0], files[1]

    def separate_data(self, original_data):
        '''

        用out-of-sample bootstrap方法产生训练集和测试集,参考论文An Empirical Comparison of Model Validation Techniques for Defect Prediction Models
        A bootstrap sample of size N is randomly drawn with replacement from an original dataset that is also of size N .
        The model is tested using the rows that do not appear in the bootstrap sample.
        On average, approximately 36.8 percent of the rows will not appear in the bootstrap sample, since the bootstrap sample is drawn with replacement.
        OriginalData:整个数据集

        return: 划分好的 训练集和测试集

        '''

        original_data = original_data.iloc[:, 3:]

        original_data = np.array(original_data)

        # 从originaldata中有放回的抽样，size(trainingdata)==size(originaldata)
        training_data = resample(original_data)

        k = len(training_data[0])
        # print('k =', k)
        # 先转换成list 在进行数据筛选

        original_data = original_data.tolist()

        training_data = training_data.tolist()

        testing_data = []

        for i in original_data:

            if i not in training_data:
                testing_data.append(i)

        testing_data = np.array(testing_data)

        # print(len(testing_data)/len(training_data))

        training_data = np.array(training_data)

        # 降序排列train_data
        training_data = sorted(
            training_data, key=lambda x: x[-1], reverse=True)
        training_data = np.array(training_data)

        training_data_X = training_data[:, 0:k - 1]

        training_data_y = training_data[:, k - 1]

        testing_data_X = testing_data[:, 0:k - 1]

        testing_data_y = testing_data[:, k - 1]

        return training_data_X, training_data_y, testing_data_X, testing_data_y

    def cross_validation(self, original_data):
        """
        Stratification 10-fold cross validation
        假设100个数据，1-20号数据有缺陷。21-100号数据没有缺陷。
        程序应该是先把100个数据中，找出哪20个有缺陷，存在一个数组A里面。哪80个没有缺陷，存在一个数组B里面。
        然后10折的第一折。从A中无放回的抽出2个，B中无放回的抽出8个。这10个数据组成一个数组fold1.
        10折的第2折 。从A中无放回的抽出2个，B中无放回的抽出8个。这10个数据组成一个数组fold2.
        ...
        10折的第10折。A中剩下的，B中剩下的，组成一个数组fold10。
        Fold2,3,4,..,10 加在一起为training_data_1 fold 1为testing_data_1
        Fold 1,3,4,…,10 加在一起为training_data_2 fold 2 为 testing_data_2
        ...
        """

        original_data = original_data.iloc[:, 3:]
        original_data = np.array(original_data)

        defect_data = []  # list A
        nondefect_data = []  # list B

        for row in original_data:
            if row[-1:] == 0:
                nondefect_data.append(row)
            else:
                defect_data.append(row)

        defect_data = np.array(defect_data)
        nondefect_data = np.array(nondefect_data)

        # 打乱数据集
        np.random.shuffle(defect_data)
        np.random.shuffle(nondefect_data)

        # 分别统计有缺陷和无缺陷的个数
        number_defect = len(defect_data)
        number_nondefect = len(nondefect_data)

        # 每次抽取的数据个数
        size_defect = int(number_defect / 10)
        size_nondefect = int(number_nondefect / 10)

        # print('defect_data =', defect_data.shape)
        # print('nondefect_data =', nondefect_data.shape)

        # print('size_defect =', size_defect)
        # print('size_nondefect =', size_nondefect)

        # print('number_defect =', number_defect)
        # print('number_nondefect =', number_nondefect)

        fold1 = np.vstack((defect_data[:size_defect],
                           nondefect_data[:size_nondefect]))
        # print('fold1 =', fold1.shape)

        fold2 = np.vstack((defect_data[size_defect:size_defect * 2],
                           nondefect_data[size_nondefect:size_nondefect * 2]))

        fold3 = np.vstack((defect_data[size_defect * 2:size_defect * 3],
                           nondefect_data[size_nondefect * 2:size_nondefect * 3]))

        fold4 = np.vstack((defect_data[size_defect * 3:size_defect * 4],
                           nondefect_data[size_nondefect * 3:size_nondefect * 4]))

        fold5 = np.vstack((defect_data[size_defect * 4:size_defect * 5],
                           nondefect_data[size_nondefect * 4:size_nondefect * 5]))

        fold6 = np.vstack((defect_data[size_defect * 5:size_defect * 6],
                           nondefect_data[size_nondefect * 5:size_nondefect * 6]))

        fold7 = np.vstack((defect_data[size_defect * 6:size_defect * 7],
                           nondefect_data[size_nondefect * 6:size_nondefect * 7]))

        fold8 = np.vstack((defect_data[size_defect * 7:size_defect * 8],
                           nondefect_data[size_nondefect * 7:size_nondefect * 8]))

        fold9 = np.vstack((defect_data[size_defect * 8:size_defect * 9],
                           nondefect_data[size_nondefect * 8:size_nondefect * 9]))

        fold10 = np.vstack((defect_data[size_defect * 9:number_defect],
                            nondefect_data[size_nondefect * 9:number_nondefect]))
        '''
        print('fold1=',fold1)
        print('fold2 =', fold2)
        print('fold3 =', fold3)
        print('fold4 =', fold4)
        print('fold5 =', fold5)
        print('fold6 =', fold6)
        print('fold7 =', fold7)
        print('fold8 =', fold8)
        print('fold9 =', fold9)
        print('fold10 =', fold10)
        '''

        # createVar = locals()
        # for i in range(10):
        #     print('i =', i)
        #     if i < 9:
        #         createVar['fold' + str(i + 1)] = np.append(defect_data[size_defect * i:size_defect * i + 1],
        #                                                    nondefect_data[size_defect * i:size_defect * i + 1])
        #     else:
        #         fold10 = np.append(defect_data[size_defect * 9:number_defect],
        #                            nondefect_data[size_defect * 9:number_nondefect])

        training_data_list = []
        testing_data_list = []

        training_data_1 = np.concatenate(
            (fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10))
        testing_data_1 = fold1
        training_data_list.append(training_data_1)
        testing_data_list.append(testing_data_1)

        training_data_2 = np.concatenate(
            (fold1, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10))
        testing_data_2 = fold2
        training_data_list.append(training_data_2)
        testing_data_list.append(testing_data_2)

        training_data_3 = np.concatenate(
            (fold1, fold2, fold4, fold5, fold6, fold7, fold8, fold9, fold10))
        testing_data_3 = fold3
        training_data_list.append(training_data_3)
        testing_data_list.append(testing_data_3)

        training_data_4 = np.concatenate(
            (fold1, fold2, fold3, fold5, fold6, fold7, fold8, fold9, fold10))
        testing_data_4 = fold4
        training_data_list.append(training_data_4)
        testing_data_list.append(testing_data_4)

        training_data_5 = np.concatenate(
            (fold1, fold2, fold3, fold4, fold6, fold7, fold8, fold9, fold10))
        testing_data_5 = fold5
        training_data_list.append(training_data_5)
        testing_data_list.append(testing_data_5)

        training_data_6 = np.concatenate(
            (fold1, fold2, fold3, fold4, fold5, fold7, fold8, fold9, fold10))
        testing_data_6 = fold6
        training_data_list.append(training_data_6)
        testing_data_list.append(testing_data_6)

        training_data_7 = np.concatenate(
            (fold1, fold2, fold3, fold4, fold5, fold6, fold8, fold9, fold10))
        testing_data_7 = fold7
        training_data_list.append(training_data_7)
        testing_data_list.append(testing_data_7)

        training_data_8 = np.concatenate(
            (fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold10))
        testing_data_8 = fold8
        training_data_list.append(training_data_8)
        testing_data_list.append(testing_data_8)

        training_data_9 = np.concatenate(
            (fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold10))
        testing_data_9 = fold9
        training_data_list.append(training_data_9)
        testing_data_list.append(testing_data_9)

        training_data_10 = np.concatenate(
            (fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9))
        testing_data_10 = fold10
        training_data_list.append(training_data_10)
        testing_data_list.append(testing_data_10)

        '''
        print('training_data_1 =', training_data_1)
        print('testing_data_1=',testing_data_1)
        
        print('training_data_list[1]',training_data_list[0])
        print('testing_data_list[1]',testing_data_list[0])
        '''

        return training_data_list, testing_data_list


'''
    def test(self):
        for dataset, filename in self.import_single_data():

            training_data_list, testing_data_list = self.cross_validation(
                dataset)
           # print('trainingdata',training_data_list)
           # print('testdata',testing_data_list)


if __name__ == '__main__':
    T = Processing()
    T.test()
'''