import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC, SVR
from collections import Counter
import pandas as pd

__author__ = 'liuqinyi'
__version__ = '0.1'

logger = logging.getLogger(__name__)


class spies:
    """
    PU spies method, based on Liu, Bing, et al. "Partially supervised classification of
    text documents." ICML. Vol. 2. 2002.
    """
    def __init__(self, first_model, second_model):
        """
        Any two models which have methods fit, predict and predict_proba can be passed,
        for example" `spies(XGBClassifier(), XGBClassifier())`
        """
        self.first_model = first_model
        self.second_model = second_model
        
    def fit(self, X, y,  spie_rate=0.2, iterate=100):
        """
        Trains models using spies method using training set (X, y).

        Parameters
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (1 for positive, 0 for unlabeled).
            
        spie_rate : {float} = 0.2 (default)
            Determines percentage of spies which will be included when training first model.
            
        spie_tolerance : {float} = 0.05 (default)
            Determines tolerated percentage of spies which can come from the first model.
            Using this tolerance threshold is chosen which splits dataset into Likely negative
            and unlabeled groups.

        Returns
        -------
        self : object
            Returns self.
        """
        np.random.seed(42)
        # 循环100次，每次使用随机的S集, 获取绝对负样
        P = X[y[0] == 1]
        U = X[y[0] == 0]
        RN_id = []
        for i in range(iterate):
            # 随机获取间谍集S
            spie_mask = np.random.random(y[0].sum()) < spie_rate
            S = P[spie_mask]
            # U+S
            US = np.vstack([U, P[spie_mask]])
            US_y = np.hstack([np.zeros((y[0] == 0).sum()), np.ones(spie_mask.sum())])
            # 为了后面更好地统计每次循环中的绝对负样，这里在y中加入一列对照id
            # US_id = np.hstack([y[1][y[0] == 0], y[1][y[0] == 1][spie_mask]])

            # P-S
            PS = P[~spie_mask]
            # print('num of P-S:', PS.shape[0], 'num of U+S:', US.shape[0])

            # 准备第一个模型的训练集
            USP = np.vstack([US, PS])
            USP_y = np.hstack([np.zeros(US.shape[0]), np.ones(PS.shape[0])])

            # Fit first model
            self.first_model.fit(X=USP, y=USP_y)

            # 确认取绝对负值的阈值tr, 由S集预测概率的10%的分为数决定
            S_prob = self.first_model.predict_proba(S)
            S_prob = S_prob[:, 1]
            tr = np.percentile(S_prob, 10)

            U_prob = self.first_model.predict_proba(U)
            U_prob = U_prob[:, 1]
            # 得到一次循环的负样的ID
            N_id = y[1][y[0] == 0][U_prob <= tr]
            RN_id = RN_id + N_id.tolist()

        # 统计在iter次循环中都出现的负样即为绝对负样
        RN_dict = Counter(RN_id)
        RN_id_list = []
        for (K, V) in RN_dict.items():
            if V == iterate:
                RN_id_list.append(K)
        RN_id_list = np.array(RN_id_list)

        # 得到绝对负样的特征集
        RN = X[(y[0] == 0) & (np.isin(y[1], RN_id_list))]
        RNP = np.vstack([RN, P])
        RNP_y = np.hstack([np.zeros(RN.shape[0]), np.ones(P.shape[0])])
        print('正样数:', P.shape[0], '绝对负样数:', RN.shape[0])

        # Fit second model
        X_train, X_test, y_train, y_test = train_test_split(RNP, RNP_y, test_size=0.2)
        self.second_model.fit(X_train, y_train)
        print('score:', self.second_model.score(X_test, y_test))

        # 交叉验证
        cross_scores = cross_val_score(self.second_model, RNP, RNP_y, cv=5).mean()
        print('cross scores:', cross_scores)
        y_predict = self.second_model.predict(RNP)
        print('confusion matrix:', confusion_matrix(y_true=RNP_y, y_pred=y_predict))

        # Unknown 预测结果
        unknown = X[(y[0] == 0) & (np.isin(y[1], RN_id_list, invert=True))]
        unknown_predict = self.second_model.predict(unknown)
        print('Predict of Real Unknown:%d/%d' % (unknown_predict.sum(), unknown.shape[0]))

        # 计算最后一次S被判定正样的结果
        S_predict = self.second_model.predict(S)
        print('Predict of S:%d/%d' % (S_predict.sum(), S.shape[0]))

    def predict(self, X):
        """
        Predicts classes for X. Uses second trained model from self.

        Parameters
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        return self.second_model.predict(np.array(X))
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X. Uses second trained model from self.

        Parameters
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples.
        """
        return self.second_model.predict_proba(np.array(X))[:,1]


def load_disease_embedding(embedding_file_name):
    with open(embedding_file_name) as f:
        node_num, emb_size = f.readline().split()
        print('Nodes with embedding: %s' % node_num)
        embedding_look_up = {}
        for line in f:
            vec = line.strip().split()
            node_id = vec[0]
            emb = [float(x) for x in vec[1:]]
            emb = emb / np.linalg.norm(emb)
            emb[np.isnan(emb)] = 0
            embedding_look_up[node_id] = np.array(emb)
    f.close()
    return embedding_look_up


def main(embedding_file_name, disease_label_file):
    # 读取label 文件，两列(id , label("seed", "unknown"))
    disease_id_label = pd.read_csv(disease_label_file, sep='\t', header=0)

    # 读取embedding文件
    embedding_look_up = load_disease_embedding(embedding_file_name)

    # 获得X(shape=[n_sample, n_features]) Y(shape=[label("0","1"), id])
    X = np.array([embedding_look_up[str(inx)] for inx in disease_id_label['id']])
    Y = disease_id_label['label'].map(lambda x: 1 if x == 'seed' else 0).values
    ID = disease_id_label['id'].values
    Y = np.stack([Y, ID], axis=0)

    # 训练Pulearning 对象，这里使用spise方法，后续可以加入其它两种方法
    model = spies(GaussianNB(), SVC(kernel='linear', gamma='auto'))
    model.fit(X, Y, spie_rate=0.2, iterate=10)


if __name__ == '__main__':
    main('../../datasets/disease/xinjibing/DeepWalk.txt', '../../datasets/disease/xinjibing/label.csv')
