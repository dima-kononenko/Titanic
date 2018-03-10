import helpers as h
import matplotlib.pyplot as plt
import sys
import sklearn.metrics as sm


def train_with_dummy(args):
    train_params, train_df = h.clean_with_dummy(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    classifier, test_costs, n = h.train_knn(X_train, Y_train, X_test, Y_test)
    Y_test_pred = classifier.predict(X_test)
    metrics = {
                'accuracy': {'train': classifier.score(X_train, Y_train), 'test': classifier.score(X_test, Y_test)},
                'confusion_matrix': sm.confusion_matrix(Y_test, Y_test_pred),
                'classification_report': sm.classification_report(Y_test, Y_test_pred)
            }
    return train_params, metrics, classifier, test_costs, n

def train_with_mapping(args):
    train_params, train_df = h.clean_with_mapping(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    classifier, test_costs, n = h.train_knn(X_train, Y_train, X_test, Y_test)
    Y_test_pred = classifier.predict(X_test)
    metrics = {
                'accuracy': {'train': classifier.score(X_train, Y_train), 'test': classifier.score(X_test, Y_test)},
                'confusion_matrix': sm.confusion_matrix(Y_test, Y_test_pred),
                'classification_report': sm.classification_report(Y_test, Y_test_pred)
            }
    return train_params, metrics, classifier, test_costs, n

if __name__ == "__main__":
    args = sys.argv[1:]
    printstats = bool(args[3]) if len(args) > 3 else False
    
    dummy_params, dummy_metrics, dummy_knn, dummy_costs, dummy_n = train_with_dummy(args)
    mapping_params, mapping_metrics, mapping_knn, mapping_costs, mapping_n = train_with_mapping(args)

    if printstats:
        print ("Train Accuracy With Dummy (n = {0}):{1}".format(dummy_n, dummy_metrics['accuracy']['train']))
        print ("Test Accuracy With Dummy (n = {0}):{1}".format(dummy_n, dummy_metrics['accuracy']['test']))
        print ("Confusion Matrix With Dummy (n = {0}):\n{1}".format(dummy_n, dummy_metrics['confusion_matrix']))
        print ("Classification Report With Dummy (n = {0}):\n{1}".format(dummy_n, dummy_metrics['classification_report']))
        print ("Train Accuracy With Mapping (n = {0}):{1}".format(mapping_n, mapping_metrics['accuracy']['train']))
        print ("Test Accuracy With Mapping (n = {0}):{1}".format(mapping_n, mapping_metrics['accuracy']['test']))
        print ("Confusion Matrix With Mapping (n = {0}):\n{1}".format(mapping_n, mapping_metrics['confusion_matrix']))
        print ("Classification Report With Mapping (n = {0}):\n{1}".format(mapping_n, mapping_metrics['classification_report']))
        plt.figure()
        plt.plot(dummy_costs, label='dummy')
        plt.plot(mapping_costs, label='mapping')
        plt.legend()
        plt.figure()
        h.plot_confusion_matrix(dummy_metrics['confusion_matrix'], "Confusion Matrix With Dummy")
        plt.figure()
        h.plot_confusion_matrix(mapping_metrics['confusion_matrix'], "Confusion Matrix With Mapping")
        
    if dummy_metrics['accuracy']['test'] > mapping_metrics['accuracy']['test']:
        print("Predicting with dummy variables")
        _, df = h.clean_with_dummy(args[1] if len(args) > 1 else 'test.csv', dummy_params)
        df = df.drop(['Survived'], axis = 1)
        df = h.predict_knn(df, dummy_knn)
    else:
        print("Predicting with mapping")
        _, df = h.clean_with_mapping(args[1] if len(args) > 1 else 'test.csv', mapping_params)
        df = h.predict_knn(df, mapping_knn)
        
    df['Survived'].astype('int64').to_csv(args[2] if len(args) > 2 else 'predict.csv', header = True)


