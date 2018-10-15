import pandas as pd
import numpy as np
import scipy as scp

# column size of input data must be 3 (base_ids, target_ids, rating)
class GroupLens():
    def __init__(self, sim_type='corr', min_pair_rating=5):
        assert sim_type in ['corr', 'cosine'], 'sim_type must be one of ["corr", "cosine"]'
        assert type(min_pair_rating) is int, 'min_pair_rating must be an integer'
        self._sim_type_dict = dict({'corr': 'pearson_corr', 'cosine': 'cosine_similarity'})
        self.sim_type = self._sim_type_dict[sim_type]
        self.min_pair_rating = min_pair_rating

    def _make_ave_score_dict(self, base_feat_vec):
        ''' dictionary for average score '''
        ave_score_dict = dict()
        for uid in self.uid_ls:
            ave_score = base_feat_vec.loc[uid].dropna().mean()
            ave_score_dict.update({uid: ave_score})
        return ave_score_dict

    def _make_weight_dict(self, base_feat_vec):
        ''' dictionary for weight '''
        weight_dict = dict()
        for uid1 in self.uid_ls:
            single_weight_dict = dict()
            for uid2 in self.uid_ls:
                pair_rating = base_feat_vec.loc[[uid1, uid2]].T.dropna()
                # if number of co-rated movies are smaller than min_pair_rating, weight is not calculated
                if len(pair_rating) < self.min_pair_rating:
                    continue
                if self.sim_type == 'pearson_corr':
                    weight = pair_rating.corr().iloc[0, 1]
                elif self.sim_type == 'cosine_similarity':
                    weight = 1 - scp.spatial.distance.cosine(
                        pair_rating.iloc[:, 0], pair_rating.iloc[:, 1])
                single_weight_dict.update({uid2: weight})
            weight_dict.update({uid1: single_weight_dict})
        return weight_dict

    def fit(self, X, y):
        '''training'''
        train_ratings = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        train_ratings.columns = ['base_id', 'target_id', 'interaction']
        self.user_item_vec = train_ratings.pivot_table(
                index='base_id', columns='target_id', values='interaction')
        self.uid_ls = list(self.user_item_vec.index)
        self.ave_score_dict = self._make_ave_score_dict(self.user_item_vec)
        self.weight_dict = self._make_weight_dict(self.user_item_vec)
        self.comb_weight_flag = False

    def fit_comb_weight(self, X, y, content_vec):
        '''training when using linear combination weight'''
        self.fit(X, y)
        self.base_feat_weight_dict = self._make_weight_dict(content_vec)
        self.comb_weight_flag = True

    def predict_score(self, uid, mid, comb_weight_flag=False, comb_weight=(0.7, 0.3)):
        '''predict rating of single item (movie) from single user'''
        w1, w2 = comb_weight
        ave_score1 = self.ave_score_dict[uid]
        valid_uid_score = self.user_item_vec.loc[:, mid].dropna()
        numerator = 0
        denominator = 0
        for uid2, score2 in valid_uid_score.iteritems():
            ave_score2 = self.ave_score_dict[uid2]
            weight = self.weight_dict[uid][uid2]
            if comb_weight_flag:
                # linear combination of weight
                weight2 = self.base_feat_weight_dict[uid][uid2]
                weight = (weight*w1 + weight2*w2)/(w1 + w2)
            denominator += abs(weight)
            numerator += weight*(score2 - ave_score2)
        pre_score = ave_score1 + (numerator / denominator)
        return pre_score

    def predict(self, X, comb_weight_flag=None, comb_weight=(0.7, 0.3)):
        '''predict ratings of multiple items from multiple users'''
        if comb_weight_flag is not None:
            self.comb_weight_flag = comb_weight_flag

        self.err_ls = list()
        result_ls = list()
        for uid, mid in X.values:
            try:
                score = self.predict_score(uid, mid, self.comb_weight_flag, comb_weight)
                result_ls.append((uid, mid, score))
            except KeyError:
                # when weight for (uid, uid2) does not exist, incorporate pad with average
                result_ls.append((uid, mid, self.ave_score_dict[uid]))
                self.err_ls.append((uid, mid))
        prediction = pd.DataFrame(np.array(result_ls), columns=[list(X.columns) + ['prediction']])
        return prediction
