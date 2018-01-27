import helpers as h
import matplotlib.pyplot as plt
import sys


def train_with_dummy(args):
    train_params, train_df = h.clean_with_dummy(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    model_params, costs, accuracy = h.train_logistic(X_train, Y_train, X_test, Y_test, False)
    train_params.update(model_params)
    return train_params, accuracy, costs

def train_with_mapping(args):
    train_params, train_df = h.clean_with_mapping(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    model_params, costs, accuracy = h.train_logistic(X_train, Y_train, X_test, Y_test, False)
    train_params.update(model_params)
    return train_params, accuracy, costs

if __name__ == "__main__":
    args = sys.argv[1:]
    printstats = bool(args[3]) if len(args) > 3 else False
    
    dummy_params, dummy_accuracy, dummy_costs = train_with_dummy(args)
    mapping_params, mapping_accuracy, mapping_costs = train_with_mapping(args)

    if printstats:
        print ("Train Accuracy With Dummy:", dummy_accuracy['train'])
        print ("Test Accuracy With Dummy:", dummy_accuracy['test'])
        print ("Train Accuracy With Mapping:", mapping_accuracy['train'])
        print ("Test Accuracy With Mapping:", mapping_accuracy['test'])
        plt.plot(dummy_costs, label = 'dummy')
        plt.plot(mapping_costs, label = 'mapping')
        plt.legend()

    if dummy_accuracy['test'] > mapping_accuracy['test']:
        print("Predicting with dummy variables")
        _, df = h.clean_with_dummy(args[1] if len(args) > 1 else 'test.csv', dummy_params)
        df = df.drop(['Survived'], axis = 1)
        df = h.predict_logistic(df, dummy_params)
    else:
        print("Predicting with mapping")
        _, df = h.clean_with_mapping(args[1] if len(args) > 1 else 'test.csv', mapping_params)
        df = h.predict_logistic(df, mapping_params)
        
    df['Survived'].astype('int64').to_csv(args[2] if len(args) > 2 else 'predict.csv', header = True)

