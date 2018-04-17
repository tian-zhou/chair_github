"""
-------------------------------------------------------------------------------
Name:        test module
Purpose:     test purpose
Idea:        how to solve it
Time:        N/A
Space:       N/A
Author:      Tian Zhou
Email:       zhou338 [at] purdue [dot] edu
Created:     01/12/2015
Copyright:   (c) Tian Zhou 2015
-------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut

def load_relevant_files():
    feature_names = list(pd.read_excel('../annotation/column_names.xlsx',
                        sheetname='descriptor')['Name'])
    selected_feature_names = list(pd.read_excel('../annotation/column_names.xlsx',
                        sheetname='selected')['Name'])
    df_req_all = pd.read_pickle('../data_proc/segmented/df_req_all.pkl')
    df_req_all['y'] = 1
    df_ope_all = pd.read_pickle('../data_proc/segmented/df_ope_all.pkl')
    df_ope_all['y'] = 0
    df = df_req_all.append(df_ope_all, ignore_index=True)
    return df, feature_names, selected_feature_names

def visualize_segments():
    df, feature_names, selected_feature_names = load_relevant_files()
    for feat_idx, feature in enumerate(selected_feature_names):
        print "Visualize feature %s" % feature
        plt.figure(figsize=(12, 8))
        for subj_idx, subject in enumerate([1,2,3,5,6,7,8]):
            df_subject = df.loc[df['subjID'] == subject]

            for idx in df_subject.index:
                row = df_subject.loc[idx]
                line = 'r-' if row['y'] == 1 else 'g-'
                plt.subplot(2,4,subj_idx+1)
                plt.plot(row['mm_data']['realtime_obj'], row['mm_data'][feature], line)
                plt.xticks([])

        plt.suptitle(feature)
        plt.show()
##        plt.savefig('../figures/each_feature/'+str(feat_idx)+'_'+feature+'.png')

def analyze():
    df, feature_names, selected_feature_names = load_relevant_files()
    for feature in selected_feature_names:
##        if feature != 'Myo_yaw':
##            continue

        X = np.zeros((0,1))
        for i, subject in enumerate([1,2,3,5,6,7,8]):
            df_subject = df.loc[df['subjID'] == subject]
            for idx in df_subject.index:
                row = df_subject.loc[idx]
                X = np.concatenate((X, row['mm_data'][feature].values.reshape(-1,1)))

        # run some statistis
##        N = len(X)
##        outlier_option = 'boarder' # ['IQR', 'boarder']
##        if outlier_option == 'boarder':
##            r1 = np.percentile(X, 1)
##            r99 = np.percentile(X, 99)
##            upper_fence = r99
##            lower_fence = r1
##        elif outlier_option == 'IQR':
##            r25 = np.percentile(X, 25)
##            r75 = np.percentile(X, 75)
##            IQR = r75-r25
##            multi = 3
##            upper_fence = r75 + multi * IQR
##            lower_fence = r25 - multi * IQR
##        upper_outlier_ratio = np.sum(X > upper_fence) / float(N)
##        lower_outlier_ratio = np.sum(X < lower_fence) / float(N)
##        print "(%.3f, %.3f) <- feature %s " % (upper_outlier_ratio, lower_outlier_ratio, feature)

def main():
    visualize_segments()

if __name__ == '__main__':
    main()