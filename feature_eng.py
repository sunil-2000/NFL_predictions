import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


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
        self.X, self.Y = data_matrix[:, :-1], data_matrix[:, -1]
        self.x_tr, self.y_tr, self.x_te, self.y_te, self.idx_tr, self.idx_te = Data.train_test_split(data_matrix)

    @staticmethod
    def train_test_split(data):
        x, y = data[:, :-1], data[:, -1]
        idx = np.arange(data.shape[0]) 
        x_tr, x_te, y_tr, y_te, idx_tr, idx_te = train_test_split(x, y, idx, test_size=0.25)
        return x_tr, y_tr, x_te, y_te, idx_tr, idx_te

    def lr(self, normalize=False):
        assert self.x_tr is not None
        # logistic regression requires normalizing data
        x_tr = self.x_tr
        if normalize:
            scaler = MinMaxScaler()
            x_tr = scaler.fit_transform(self.x_tr)
        model = LogisticRegression().fit(x_tr, self.y_tr)
        return model
    
    def evaluate_test(self, model, normalize=False):
        if normalize:
            scaler = MinMaxScaler()
            x_te = scaler.fit_transform(self.x_te)
        else:
            x_te = self.x_te
        y_pred = model.predict(x_te)
        accuracy = accuracy_score(self.y_te, y_pred)
        print(f'accuracy: {accuracy}')
        return accuracy

    def evaluate_train(self, model):
        y_pred = model.predict(self.x_tr)
        accuracy = accuracy_score(self.y_tr, y_pred)
        print(f'accuracy: {accuracy}')
        return accuracy

    def rf(self):
        model = RandomForestClassifier(n_estimators = 100)
        model = model.fit(self.x_tr, self.y_tr)
        return model

    def neural_net(self):
        model = MLPClassifier(hidden_layer_sizes=(10,10), random_state=1)
        model = model.fit(self.x_tr, self.y_tr)
        return model
    
    def single_prediction(self, model):
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


    def kfold_validation(self, model_lst, normalize=False):
        X, Y = self.X, self.Y
        if normalize:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(self.X)
        kf = StratifiedKFold(shuffle=True)
        out = {}
        for model in model_lst:
            model_name = type(model).__name__
            scores = cross_val_score(
            model, X, Y, scoring='accuracy', cv=kf, n_jobs=-1)
            print(model_name)
            out[model_name] = scores
        return out
