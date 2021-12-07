from networks.ANN import ANN
from train_evaluate.train_and_evaluation_numpy_ann import train_and_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ann = ANN()
    train_and_test(ann)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
