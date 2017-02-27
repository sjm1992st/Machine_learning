import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import import_hw as dat


m_data_x,m_data_y=dat.import_data()
x = m_data_x
y=m_data_y.ravel()
#print y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

h = .02
# create a mesh to plot in
x_min, x_max = x_train[:, 0].min() - 0.1, x_train[:, 0].max() + 0.1
y_min, y_max = x_train[:, 1].min() - 0.1, x_train[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

''''' SVM '''
# title for the plots
titles = ['LinearSVC (linear kernel)',
          'SVC with polynomial (degree 3) kernel',
          'SVC with RBF kernel',
          'SVC with Sigmoid kernel']
clf_linear  = svm.SVC(kernel='linear').fit(x, y)
#clf_linear  = svm.LinearSVC().fit(x, y)
clf_poly    = svm.SVC(kernel='poly', degree=3).fit(x, y)
clf_rbf     = svm.SVC().fit(x, y)
clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x, y)

for i, clf in enumerate((clf_linear, clf_poly, clf_rbf)):    ##clf_sigmoid
    answer = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    y_pred=clf.predict(x_test)
    #answer = clf.predict(y_test, y_pred)
    print f1_score(y_test, y_pred)
    """
    print(clf)
    print(np.mean( answer == y_train))
    print(answer)
    print(y_train)
    """
    plt.figure(figsize=(12,8))
    #plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Put the result into a color plot
    precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
    #print classification_report(y_test, y_pred, target_names = ['-1', '1'])
    #answer = clf.predict(np.c_[xx.ravel(), yy.ravel()])[:,1]
    z = answer.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title(titles[i])
plt.show()