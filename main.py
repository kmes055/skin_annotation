import numpy as np
from BayesianClassifier import BayesianClassifier
from GaussianClassifier import GaussianClassifier
from time import time
import part_B, part_C

prior_bayesian = [x / 100 for x in range(35, 50)]
coeff_gaussian = [0.02, 0.04, 0.06, 0.08, 0.1]


def cross_validation(model, data, k=5):
    # Shuffle data
    np.random.seed(np.random.randint(time()))
    shuffled_index = np.random.permutation(data.shape[0])
    data = data[shuffled_index]
    fold = data.shape[0] // k
    scores = []
    models = []

    for i in range(k):
        train = np.vstack((data[:i * fold, :], data[(i + 1) * fold:, :]))
        test = data[i * fold:(i + 1) * fold]
        X_train, y_train = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]

        model.fit(X_train, y_train)
        models.append(model)

        precision, recall = model.evaluate(X_test, y_test)
        # print('Precision :\t%.4f\nRecall :\t%.4f' % (precision, recall))
        # F1 score : 2/S = 1/P + 1/R
        scores.append(2*precision*recall / (precision + recall))

    print('Max score : %.4f' % np.max(scores))
    return np.max(scores), models[np.argmax(scores)]

    # print('Maximum score : %4f\n' % np.max(scores))


def hyperparameter_tuning(part, name):
    hyper = prior_bayesian if name == 'B' else coeff_gaussian
    evals = []
    best_val, best_param, best_model = 0, 0, None

    for param in hyper:
        val, model = part(param)
        evals.append(val)
        if val > best_val:
            best_val = val
            best_param = param
            best_model = model

    from matplotlib import pyplot as plt
    plt.scatter(hyper, evals)
    if name == 'G':
        plt.xlim(0, 0.12)
    plt.xticks(hyper, [str(h) for h in hyper])

    plt.show()

    return best_param, best_model


# Get data from two ways.
# Part C - 1 is in here.
b = part_B.get_data()


def b1a(prior=0.37):
    # print('Part B - 1 - a')
    clf = BayesianClassifier(only_red=True, prior=prior)
    data = b.copy()
    return cross_validation(clf, data)


def b1b(prior=0.48):
    # print('Part B - 1 - b')
    clf = BayesianClassifier(prior=prior)
    data = b.copy()
    return cross_validation(clf, data)


def b2a(prior=0.38):
    # print('Part B - 2 - a')
    clf = GaussianClassifier(only_red=True, prior=prior)
    data = b.copy()
    return cross_validation(clf, data)


def b2b(coeff=0.06):
    # print('Part B - 2 - b')
    clf = GaussianClassifier(coeff=coeff)
    data = b.copy()
    return cross_validation(clf, data, k=5)


def b3():
    val_b1a, model_b1a = hyperparameter_tuning(b1a, 'B')
    print('Best parameter of B - 1 - a is %f' % val_b1a)
    val_b1b, model_b1b = hyperparameter_tuning(b1b, 'B')
    print('Best parameter of B - 1 - b is %f' % val_b1b)
    val_b2a, model_b2a = hyperparameter_tuning(b2a, 'B')
    print('Best parameter of B - 2 - a is %f' % val_b2a)
    val_b2b, model_b2b = hyperparameter_tuning(b2b, 'G')
    print('Best parameter of B - 2 - b is %f' % val_b2b)


def c1():
    # print('Part C - 1')
    # c, img_dic = part_C.get_data()  # for fast execution.
    c, img_dic = part_C.create_data()
    print('Head of data')
    print(c[:100])

    return c, img_dic


def c2(coeff=0.06):
    # print('Part C - 2')
    # best_val, best_model = 0, None
    data = c.copy()
    val, model = cross_validation(GaussianClassifier(coeff=0.06), data)

    part_C.predict_image(model, img_dic)


# print('Part B. Skin detection using text data')
# print('1. (a)')
# b1a()
# print('1. (b)')
# b1b()

# print('2 .(a)')
# b2a()
# print('2. (b)')
# b2b()

# print('3. Hyper-parameter tuning.')
# b3()

# print('Part C. Skin detection using real data.')
# print('Warning!! this part will take many memory and time.')
# print('1. ')
# c, img_dic = c1()
# print('2.')
# c2()
