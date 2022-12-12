import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


class Data():
    def __init__(self, csv, classification=True, playoff_encoded=False):
        """
        csv: path to csv
        col_lst: list of columns to use
        classification(bool): whether classification model
        """
        if not classification:
            raise NotImplementedError
        self.data = pd.read_csv(csv)
        self.home_score = list(self.data['home_score'])
        self.away_score = list(self.data['away_score'])
        if classification:
            # only concerned about whether home team won or not
            self.data = self.data.drop(['home_score', 'away_score'], axis=1)
        self.date = list(self.data['date_x'])
        self.home_teams = list(self.data['home_team']) # for outputing prediction
        self.away_teams = list(self.data['away_team'])
        #ensure numeric
        ftrs = ['playoff', 'elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'qbelo1_pre', 
          'qbelo2_pre', 'qb1_value_pre', 'qb2_value_pre', 'qb1_adj', 
          'qb2_adj', 'qbelo_prob1', 'qbelo_prob2', 'quality', 'Temperature', 
          'DewPoint', 'Humidity', 'Precipitation', 'WindSpeed', 'Pressure']
        if playoff_encoded:
            ftrs = ['Playoff__c', 'Playoff__d', 'Playoff__n', 'Playoff__s', 'Playoff__w',
              'elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'qbelo1_pre',
              'qbelo2_pre', 'qb1_value_pre', 'qb2_value_pre', 'qb1_adj',
              'qb2_adj', 'qbelo_prob1', 'qbelo_prob2', 'quality', 'Temperature',
              'DewPoint', 'Humidity', 'Precipitation', 'WindSpeed', 'Pressure']
        data_matrix = np.array(self.data[ftrs+['home_win']])
        self.x_tr, self.y_tr, self.x_te, self.y_te, self.idx_tr, self.idx_te = Data.train_test_split(data_matrix)

    @staticmethod
    def train_test_split(data):
        x, y = data[:, :-1], data[:, -1]
        idx = np.arange(data.shape[0]) 
        x_tr, x_te, y_tr, y_te, idx_tr, idx_te = train_test_split(x, y, idx, test_size=0.25)
        return x_tr, y_tr, x_te, y_te, idx_tr, idx_te

    def lr(self):
        assert self.x_tr is not None
        # logistic regression requires normalizing data
        scaler = MinMaxScaler()
        x_tr = scaler.fit_transform(self.x_tr)
        model = LogisticRegression().fit(x_tr, self.y_tr)
        return model
    
    def lr_evaluate_test(self, model):
        scaler = MinMaxScaler()
        x_te = scaler.fit_transform(self.x_te)
        y_pred = model.predict(x_te)
        accuracy = accuracy_score(self.y_te, y_pred)
        print(f'accuracy: {accuracy}')
        return accuracy

    def evaluate_train(self, model):
        y_pred = model.predict(self.x_tr)
        accuracy = accuracy_score(self.y_tr, y_pred)
        print(f'accuracy: {accuracy}')
        return accuracy

    def rf_evaluate_test(self, model):
        y_pred = model.predict(self.x_te)
        accuracy = accuracy_score(self.y_te, y_pred)
        print(f'accuracy: {accuracy}')
        return accuracy

    def random_forest(self):
        model = RandomForestClassifier(n_estimators = 100)
        model = model.fit(self.x_tr, self.y_tr)
        return model
    
    def lr_single_prediction(self, model):
        """
        random prediction with logistic regression
        show probability output
        """
        scaler = MinMaxScaler()
        i = np.random.randint(0, len(self.x_te))
        x_te = self.x_te[i,:].reshape(1,-1)
        x_te = scaler.fit_transform(x_te)
        y_pred = model.predict(x_te)
        y_prob_pred = model.predict_proba(x_te)

        og_idx = self.idx_te[i]
        date = self.date[og_idx]
        home, away = self.home_teams[og_idx], self.away_teams[og_idx]
        h_score, a_score = self.home_score[og_idx], self.away_score[og_idx]
        outcome = self.y_te[i]

        print(f'{date}: {home} vs. {away}')
        print(f'lr model predicts {home if y_pred == 1 else away} will win')
        print(f'with probability {y_prob_pred[0][int(y_pred)]}')
        print(f'actual game outcome: {home if outcome > 0 else away} won')
        print(f'actual score: {home}: {h_score} {away}: {a_score}')

    def rf(self):
        pass

    def kfold_validation(self):
        raise NotImplementedError

    def fit(self):
        pass

    def predict(self):
        pass

