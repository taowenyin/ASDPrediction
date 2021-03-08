import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# https://www.cnblogs.com/cafe3165/p/9145427.html

if __name__ == '__main__':
    pd.set_option('display.max_rows', 1000, 'display.max_columns', 1000, "display.max_colwidth", 1000, 'display.width',
                  1000)
    np.set_printoptions(threshold=np.inf)

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

    # 分割数据集
    train_data, test_data, train_labels, test_labels = train_test_split(asd_data, asd_target, test_size=0.2, shuffle=True)

    # 拟合数据
    clf = SVC(kernel='linear', C=0.4, gamma='auto')
    clf.fit(train_data, train_labels)

    # 预测数据
    pred_labels = clf.predict(test_data)

    # 打印结果
    print(classification_report(test_labels, pred_labels))
