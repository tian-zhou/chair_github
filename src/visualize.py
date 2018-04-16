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

def vis_entire_chunk():
    feature_names = list(pd.read_excel('../annotation/column_names.xlsx',
                        sheetname='descriptor')['Name'])
    selected_feature_names = list(pd.read_excel('../annotation/column_names.xlsx',
                        sheetname='selected')['Name'])
    fname = '../data/Multimodal_log/4_1.txt'
    df = pd.read_csv(fname, sep=' ', header=None)
    df.columns = feature_names

    for feature in selected_feature_names:
        plt.figure()
        if feature in ['Kinect_face_roll', 'Kinect_face_pitch', 'Kinect_face_yaw']:
            df[feature] = df[feature].clip(lower=-100, upper = 100)
        plt.plot(df[feature])
        plt.title(feature)
    plt.show()

def vis_segments():
    feature_names = list(pd.read_excel('../annotation/column_names.xlsx',
                        sheetname='descriptor')['Name'])
    selected_feature_names = list(pd.read_excel('../annotation/column_names.xlsx',
                        sheetname='selected')['Name'])

    for subject in [1,2,3,5,6,7,8]:
        print "Look at subject %s's segmented data" % subject
        df_req_all = pd.read_pickle('../data_proc/segmented/'+str(subject)+'_df_req.pkl')
        df_ope_all = pd.read_pickle('../data_proc/segmented/'+str(subject)+'_df_ope.pkl')

        for feature in selected_feature_names:
            plt.figure()
            for idx in df_req_all.index:
                row = df_req_all.loc[idx]
                realtime_str = row['mm_data']['realtime_str']
                realtime_obj = pd.to_datetime(realtime_str, format='%Y_%m_%d_%H_%M_%S_%f')
                plt.plot(realtime_obj, row['mm_data'][feature], 'r-')

            for idx in df_ope_all.index:
                row = df_ope_all.loc[idx]
                realtime_str = row['mm_data']['realtime_str']
                realtime_obj = pd.to_datetime(realtime_str, format='%Y_%m_%d_%H_%M_%S_%f')
                plt.plot(realtime_obj, row['mm_data'][feature], 'g-')

            plt.title(feature)
            plt.savefig('../figures/raw/'+feature+'_subj'+str(subject)+'.png')

if __name__ == '__main__':
    vis_segments()


