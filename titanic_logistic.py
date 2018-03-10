import helpers as h
import matplotlib.pyplot as plt
import sys
import sklearn.metrics as sm

def train_with_dummy(args, threshold = 0.8):
    train_params, train_df = h.clean_with_dummy(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    model_params, costs, accuracy, Y_test_pred = h.train_logistic(X_train, Y_train, X_test, Y_test, threshold, False)
    train_params.update(model_params)
    metrics = {
                'accuracy': accuracy,
                'confusion_matrix': sm.confusion_matrix(Y_test, Y_test_pred),
                'classification_report': sm.classification_report(Y_test, Y_test_pred)
            }
    return train_params, metrics, costs

def train_with_mapping(args, threshold = 0.8):
    train_params, train_df = h.clean_with_mapping(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    model_params, costs, accuracy, Y_test_pred = h.train_logistic(X_train, Y_train, X_test, Y_test, threshold, False)
    train_params.update(model_params)
    metrics = {
                'accuracy': accuracy,
                'confusion_matrix': sm.confusion_matrix(Y_test, Y_test_pred),
                'classification_report': sm.classification_report(Y_test, Y_test_pred)
            }
    return train_params, metrics, costs

if __name__ == "__main__":
    args = sys.argv[1:]
    printstats = bool(args[3]) if len(args) > 3 else False
    threshold = 0.8
    
    dummy_params, dummy_metrics, dummy_costs = train_with_dummy(args, threshold)
    mapping_params, mapping_metrics, mapping_costs = train_with_mapping(args, threshold)

    if printstats:
        print ("Train Accuracy With Dummy:", dummy_metrics['accuracy']['train'])
        print ("Test Accuracy With Dummy:", dummy_metrics['accuracy']['test'])
        print ("Confusion Matrix With Dummy:")
        print (dummy_metrics['confusion_matrix'])
        print ("Classification Report With Dummy:")
        print(dummy_metrics['classification_report'])
        print ("Train Accuracy With Mapping:", mapping_metrics['accuracy']['train'])
        print ("Test Accuracy With Mapping:", mapping_metrics['accuracy']['test'])
        print ("Confusion Matrix With Mapping:")
        print (mapping_metrics['confusion_matrix'])
        print ("Classification Report With Mapping:")
        print(mapping_metrics['classification_report'])
        plt.figure()
        plt.plot(dummy_costs, label = 'dummy')
        plt.plot(mapping_costs, label = 'mapping')
        plt.legend()
        plt.figure()
        h.plot_confusion_matrix(dummy_metrics['confusion_matrix'], "Confusion Matrix With Dummy")
        plt.figure()
        h.plot_confusion_matrix(mapping_metrics['confusion_matrix'], "Confusion Matrix With Mapping")
        
    if dummy_metrics['accuracy']['test'] > mapping_metrics['accuracy']['test']:
        print("Predicting with dummy variables")
        _, df = h.clean_with_dummy(args[1] if len(args) > 1 else 'test.csv', dummy_params)
        df = df.drop(['Survived'], axis = 1)
        df = h.predict_logistic(df, dummy_params, threshold)
    else:
        print("Predicting with mapping")
        _, df = h.clean_with_mapping(args[1] if len(args) > 1 else 'test.csv', mapping_params)
        df = h.predict_logistic(df, mapping_params, threshold)
        
    df['Survived'].astype('int64').to_csv(args[2] if len(args) > 2 else 'predict.csv', header = True)

