import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import itertools

def map_category(col, mapping = None):
    categories = col.unique()
    if mapping is None:
        mapping = {categories[i] : i+1 for i in range(len(categories))}
    else:
        mapping.update({categories[i] : i+1 for i in range(len(categories)) if categories[i] not in mapping})
    return mapping, col.map(mapping)

def clean_with_mapping(filename, parameters = {}):
    df = pd.read_csv(filename, sep = ",", index_col = 0)
    df['Cabin'] = df['Cabin'].map(lambda x: str(x).split(' ')[0], na_action='ignore')
    df['CabinClass'] = df['Cabin'].map(lambda x: x[0], na_action='ignore')
    df['CabinNo'] = df['Cabin'].map(lambda x: float(x[1:]) if x[1:] != '' else '', na_action='ignore').replace('', np.nan)

    if "cabinClassMap" in parameters:
        _, cabinClassLabel = map_category(df["CabinClass"], parameters["cabinClassMap"])
    else:
        parameters["cabinClassMap"], cabinClassLabel = map_category(df["CabinClass"])
    df = df.assign(CabinClassLabel = cabinClassLabel)
    
    if "embarkedMap" in parameters:
        _, embarkedLabel = map_category(df["Embarked"], parameters["embarkedMap"])
    else:
        parameters["embarkedMap"], embarkedLabel = map_category(df["Embarked"])
    df = df.assign(EmbarkedLabel = embarkedLabel)
    
    if "sexMap" in parameters:
        _, sexLabel = map_category(df["Sex"], parameters["sexMap"])
    else:
        parameters["sexMap"], sexLabel = map_category(df["Sex"])
    df = df.assign(SexLabel = sexLabel)
    
    if "fareMean" not in parameters:
        parameters["fareMean"] = df["Fare"].mean()
    fareMean = parameters["fareMean"]
    if "fareStd" not in parameters:
        parameters["fareStd"] = df["Fare"].std()
    fareStd = parameters["fareStd"]
    df = df.assign(FareNorm = lambda x: (x["Fare"] - fareMean)/fareStd)
    df["FareNorm"] = df["FareNorm"].fillna(value = 0)
    
    if "ageMean" not in parameters:
        parameters["ageMean"] = df["Age"].mean()
    ageMean = parameters["ageMean"]
    if "ageStd" not in parameters:
        parameters["ageStd"] = df["Age"].std()
    ageStd = parameters["ageStd"]
    df = df.assign(AgeNorm = lambda x: (x["Age"] - ageMean)/ageStd)
    df["AgeNorm"] = df["AgeNorm"].fillna(value = 0)

    if "cabinNoMean" not in parameters:
        parameters["cabinNoMean"] = df["CabinNo"].mean()
    cabinNoMean = parameters["cabinNoMean"]
    if "cabinNoStd" not in parameters:
        parameters["cabinNoStd"] = df["CabinNo"].std()
    cabinNoStd = parameters["cabinNoStd"]
    df = df.assign(CabinNoNorm = lambda x: (x["CabinNo"] - cabinNoMean)/cabinNoStd)
    df["CabinNoNorm"] = df["CabinNoNorm"].fillna(value = 0)
    
    df = df.drop(["Name", "Ticket", "Embarked", "Sex", "Cabin", "Fare", "Age", 'CabinClass', 'CabinNo'], axis = 1)
    
    return parameters, df

def clean_with_dummy(filename, parameters = {}):
    df = pd.read_csv(filename, sep = ",", index_col = 0)
    df['Cabin'] = df['Cabin'].map(lambda x: str(x).split(' ')[0], na_action='ignore')
    df['CabinClass'] = df['Cabin'].map(lambda x: x[0], na_action='ignore')
    df['CabinNo'] = df['Cabin'].map(lambda x: float(x[1:]) if x[1:] != '' else '', na_action='ignore').replace('', np.nan)
    df = pd.get_dummies(df, dummy_na = True, columns = ['Pclass', 'Sex', 'Embarked', 'CabinClass'])

    if "fareMean" not in parameters:
        parameters["fareMean"] = df["Fare"].mean()
    fareMean = parameters["fareMean"]
    if "fareStd" not in parameters:
        parameters["fareStd"] = df["Fare"].std()
    fareStd = parameters["fareStd"]
    df = df.assign(FareNorm = lambda x: (x["Fare"] - fareMean)/fareStd)
    df["FareNorm"] = df["FareNorm"].fillna(value = 0)

    if "ageMean" not in parameters:
        parameters["ageMean"] = df["Age"].mean()
    ageMean = parameters["ageMean"]
    if "ageStd" not in parameters:
        parameters["ageStd"] = df["Age"].std()
    ageStd = parameters["ageStd"]
    df = df.assign(AgeNorm = lambda x: (x["Age"] - ageMean)/ageStd)
    df["AgeNorm"] = df["AgeNorm"].fillna(value = 0)

    if "cabinNoMean" not in parameters:
        parameters["cabinNoMean"] = df["CabinNo"].mean()
    cabinNoMean = parameters["cabinNoMean"]
    if "cabinNoStd" not in parameters:
        parameters["cabinNoStd"] = df["CabinNo"].std()
    cabinNoStd = parameters["cabinNoStd"]
    df = df.assign(CabinNoNorm = lambda x: (x["CabinNo"] - cabinNoMean)/cabinNoStd)
    df["CabinNoNorm"] = df["CabinNoNorm"].fillna(value = 0)

    if 'columns' not in parameters:
        df = df.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'CabinNo'], axis = 1)
        parameters['columns'] = df.columns
    for c in parameters['columns']:
        df[c] = 0 if c not in df.columns else df[c]
    df = df[parameters['columns']]
    
    return parameters, df

def prepare(df):
    X = df.drop(['Survived'], axis = 1)
    Y = df["Survived"].to_frame()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    return X_train, Y_train, X_test, Y_test

def train_logistic(X_train, Y_train, X_test, Y_test, threshold = 0.8, print_stats = False):
    costs = []
    tf.reset_default_graph() 
    with tf.Session() as sess:
        X = tf.placeholder(tf.float32, shape = [X_train.shape[1], None], name = "X")
        Y = tf.placeholder(tf.float32, shape = [Y_train.shape[1], None], name = "Y")
        W = tf.get_variable("W", [1, X_train.shape[1]], initializer = tf.zeros_initializer())
        b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())
        
        z = tf.sigmoid(tf.add(tf.matmul(W, X), b))
    
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = Y))
    
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
        init = tf.global_variables_initializer()
        sess.run(init)
    
        for epoch in range(15000):
    
            epoch_cost = 0.                       # Defines a cost related to an epoch
            _, batch_cost = sess.run([optimizer, cost], feed_dict={X: X_train.T, Y: Y_train.T})
            epoch_cost += batch_cost
    
            # Print the cost every epoch
            if epoch % 100 == 0 and print_stats:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)
     
        # Calculate the correct predictions
        p = tf.cast(z > threshold, "float")
        correct_prediction = tf.equal(p, Y)

        #save parameters
        parameters = {"W": W, "b": b}
        model_params = sess.run(parameters)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_stats= {'train':  accuracy.eval({X: X_train.T, Y: Y_train.T}), 'test': accuracy.eval({X: X_test.T, Y: Y_test.T})}
        Y_test_pred = p.eval({X: X_test.T, Y: Y_test.T})

    return model_params, costs, accuracy_stats, Y_test_pred.T

def train_knn(X_train, Y_train, X_test, Y_test):
    costs = []
    opt_cost = None
    opt_knn = None
    opt_n = None
    for n in range(1, 100):
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(X_train, Y_train['Survived'])
        s = knn.score(X_test, Y_test['Survived'])
        costs.append(s)
        if not opt_cost or s > opt_cost:
            opt_cost = s
            opt_knn = knn
            opt_n = n
    return opt_knn, costs, opt_n

def predict_logistic(df, parameters, threshold = 0.8):
    with tf.Session() as sess:
        W = tf.convert_to_tensor(parameters["W"])
        b = tf.convert_to_tensor(parameters["b"])
        X = tf.placeholder(tf.float32, shape = [df.shape[1], None], name = "X")
        
        z = tf.sigmoid(tf.add(tf.matmul(W, X), b))
        p = tf.cast(z > threshold, "float")
    
        df['Survived'] = sess.run(p, feed_dict = {X: df.T}).T
    
    return df

def predict_knn(df, classifier):
    df['Survived'] = classifier.predict(df)
    return df

def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    classes = ['Not Survived', 'Survived']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

