import pandas as pd
import numpy as np


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
    print(asd_dataset)

    print('xxx')
