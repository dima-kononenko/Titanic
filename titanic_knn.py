import helpers as h
import matplotlib.pyplot as plt
import sys


def train_with_dummy(args):
    train_params, train_df = h.clean_with_dummy(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    classifier, test_costs, n = h.train_knn(X_train, Y_train, X_test, Y_test)
    accuracy = {'train': classifier.score(X_train, Y_train), 'test': classifier.score(X_test, Y_test)}
    return train_params, accuracy, classifier, test_costs, n

def train_with_mapping(args):
    train_params, train_df = h.clean_with_mapping(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    classifier, test_costs, n = h.train_knn(X_train, Y_train, X_test, Y_test)
    accuracy = {'train': classifier.score(X_train, Y_train), 'test': classifier.score(X_test, Y_test)}
    return train_params, accuracy, classifier, test_costs, n

if __name__ == "__main__":
    args = sys.argv[1:]
    printstats = bool(args[3]) if len(args) > 3 else False
    
    dummy_params, dummy_accuracy, dummy_knn, dummy_costs, dummy_n = train_with_dummy(args)
    mapping_params, mapping_accuracy, mapping_knn, mapping_costs, mapping_n = train_with_mapping(args)

    if printstats:
        print ("Train Accuracy With Dummy (n = {0}):{1}".format(dummy_n, dummy_accuracy['train']))
        print ("Test Accuracy With Dummy (n = {0}):{1}".format(dummy_n, dummy_accuracy['test']))
        print ("Train Accuracy With Mapping (n = {0}):{1}".format(mapping_n, mapping_accuracy['train']))
        print ("Test Accuracy With Mapping (n = {0}):{1}".format(mapping_n, mapping_accuracy['test']))
        plt.plot(dummy_costs, label='dummy')
        plt.plot(mapping_costs, label='mapping')
        plt.legend()

    if dummy_accuracy['test'] > mapping_accuracy['test']:
        print("Predicting with dummy variables")
        _, df = h.clean_with_dummy(args[1] if len(args) > 1 else 'test.csv', dummy_params)
        df = df.drop(['Survived'], axis = 1)
        df = h.predict_knn(df, dummy_knn)
    else:
        print("Predicting with mapping")
        _, df = h.clean_with_mapping(args[1] if len(args) > 1 else 'test.csv', mapping_params)
        df = h.predict_knn(df, mapping_knn)
        
    df['Survived'].astype('int64').to_csv(args[2] if len(args) > 2 else 'predict.csv', header = True)


