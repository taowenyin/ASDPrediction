import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':
    pd.set_option('display.max_rows', 1000, 'display.max_columns', 1000, "display.max_colwidth", 1000, 'display.width',
                  1000)
    np.set_printoptions(threshold=np.inf)
    plt.figure(figsize=(16, 9))

    # 读取保护数据
    protect_data = pd.read_excel('data/protect_data.xlsx', sheet_name='ASD-保护因素条目',
                                 usecols=[5, 7, 17, 23, 32, 34, 36, 41], header=[0])
    protect_data['target'] = 1
    protect_data = np.array(protect_data)
    # 读取风险数据
    risk_data = pd.read_excel('data/risk_data.xlsx', sheet_name='Sheet1',
                              usecols=[5, 7, 17, 23, 32, 34, 36, 41], header=[1])
    risk_data['target'] = 0
    risk_data = np.array(risk_data)

    # 合并数据
    asd_dataset = np.vstack((protect_data, risk_data))
    # 数据标准化对象
    data_std = StandardScaler()
    # 获取数据和标签
    asd_target = asd_dataset[:, asd_dataset.shape[1] - 1]
    asd_data = data_std.fit_transform(asd_dataset[:, 0: asd_dataset.shape[1] - 1])

    # 采用过采样解决数据不平衡的问题
    smot_model = SMOTE(random_state=20)
    asd_data, asd_target = smot_model.fit_resample(asd_data, asd_target)

    # 分割数据集
    train_data, test_data, train_labels, test_labels = train_test_split(asd_data, asd_target, test_size=0.2,
                                                                        shuffle=True, random_state=20)

    # SVM拟合数据
    svm_clf = SVC(kernel='linear', C=0.4, class_weight='balanced')
    svm_clf.fit(train_data, train_labels)
    # 预测数据
    svm_pred_labels = svm_clf.predict(test_data)
    # 打印结果
    print('SVM classification report')
    print(classification_report(test_labels, svm_pred_labels))

    # 拟合数据
    logreg_clf = LogisticRegression(C=0.4, class_weight='balanced')
    logreg_clf.fit(train_data, train_labels)
    # 预测数据
    logreg_pred_labels = logreg_clf.predict(test_data)
    # 打印结果
    print('LogisticRegression classification report')
    print(classification_report(test_labels, logreg_pred_labels))

    # SVM的ROC
    SVM_FPR, SVM_recall, SVM_thresholds = roc_curve(train_labels, svm_clf.decision_function(train_data), pos_label=1)
    SVM_area = AUC(train_labels, svm_clf.decision_function(train_data))
    plt.subplot(1, 2, 1)
    plt.plot(SVM_FPR, SVM_recall, color='red', label='ROC curve (area = %0.2f)' % SVM_area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    # 为了让曲线不黏在图的边缘
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('SVM-ROC')
    plt.legend(loc="lower right")

    # LogisticRegression的ROC
    logreg_FPR, logreg_recall, logreg_thresholds = roc_curve(train_labels, logreg_clf.predict(train_data), pos_label=1)
    logreg_area = AUC(train_labels, logreg_clf.predict(train_data))
    plt.subplot(1, 2, 2)
    plt.plot(logreg_FPR, logreg_recall, color='red', label='ROC curve (area = %0.2f)' % logreg_area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    # 为了让曲线不黏在图的边缘
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('LogisticRegression-ROC')
    plt.legend(loc="lower right")

    plt.show()
