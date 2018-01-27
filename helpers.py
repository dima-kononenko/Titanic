import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def clean(filename, parameters = {}):
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
    Y = df["Survived"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    return X_train.T, Y_train.as_matrix().reshape(1, len(Y_train)), X_test.T, Y_test.as_matrix().reshape(1, len(Y_test))

def train(X_train, Y_train, X_test, Y_test, print_stats = False):
    costs = []
    tf.reset_default_graph() 
    with tf.Session() as sess:
        X = tf.placeholder(tf.float32, shape = [25, None], name = "X")
        Y = tf.placeholder(tf.float32, shape = [1, None], name = "Y")
        W = tf.get_variable("W", [1, 25], initializer = tf.zeros_initializer())
        b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())
        
        z = tf.sigmoid(tf.add(tf.matmul(W, X), b))
    
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = Y))
    
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
        init = tf.global_variables_initializer()
        sess.run(init)
    
        for epoch in range(15000):
    
            epoch_cost = 0.                       # Defines a cost related to an epoch
            _, batch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            epoch_cost += batch_cost
    
            # Print the cost every epoch
            if epoch % 100 == 0 and print_stats:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)
     
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.cast(z > 0.8, "float"), Y)

        #save parameters
        parameters = {"W": W, "b": b}
        model_params = sess.run(parameters)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_stats= {'train':  accuracy.eval({X: X_train, Y: Y_train}), 'test': accuracy.eval({X: X_test, Y: Y_test})}

    return model_params, costs, accuracy_stats

def predict(filename, parameters):
    _, df = clean(filename, parameters)
    df = df.drop(['Survived'], axis = 1)
    with tf.Session() as sess:
        W = tf.convert_to_tensor(parameters["W"])
        b = tf.convert_to_tensor(parameters["b"])
        X = tf.placeholder(tf.float32, shape = [25, None], name = "X")
        
        z = z = tf.sigmoid(tf.add(tf.matmul(W, X), b))
        p = tf.cast(z > 0.8, "float")
    
        df['Survived'] = sess.run(p, feed_dict = {X: df.T}).T
    
    return df