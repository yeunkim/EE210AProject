# -*- coding: utf-8 -*-
"""
Prepare metadata data.
"""

import pandas as pd

class prepare_metadata(object):

    def __init__(self, csv, csv_measures, data):
        self.demog = ''
        self.measures = ''
        self.combined_df = ''

        self.org_demog(csv, csv_measures, data)

    def org_demog(self, csv, csv_measures, data):
        self.demog = pd.read_csv(csv)
        self.measures = pd.read_csv(csv_measures)
        self.combined_df = pd.concat([self.demog, self.measures], axis=1)

        self.Y_train_demog = self.combined_df[self.combined_df['X_id'].isin(data.y_train0_id)].reset_index()
        self.Y_test_demog = self.combined_df[self.combined_df['X_id'].isin(data.y_test0_id)].reset_index()

        self.Y_train_demog = self.Y_train_demog.drop(['index'], axis=1)
        self.Y_test_demog = self.Y_test_demog.drop(['index'], axis=1)




