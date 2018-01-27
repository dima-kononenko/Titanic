import helpers as h
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    args = sys.argv[1:]
    train_params, train_df = h.clean(args[0] if len(args) > 0 else "train.csv")
    X_train, Y_train, X_test, Y_test = h.prepare(train_df)
    printstats = bool(args[3]) if len(args) > 3 else False
    model_params, costs, accuracy = h.train(X_train, Y_train, X_test, Y_test, printstats)
    train_params.update(model_params)
    if printstats:
        print ("Train Accuracy:", accuracy['train'])
        print ("Test Accuracy:", accuracy['test'])
        plt.plot(costs)
    
    df = h.predict(args[1] if len(args) > 1 else 'test.csv', train_params)
    df['Survived'].astype('int64').to_csv(args[2] if len(args) > 2 else 'predict.csv', header = True)

