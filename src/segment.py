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
from datetime import datetime, timedelta
import os
import numpy as np

class ReadMultiModal():
    def __init__(self):
        self.REF_TIME = datetime.strptime("2018_03_01_00_00_00_000", '%Y_%m_%d_%H_%M_%S_%f')

    def read(self, filename):
        """
        Description:
            Read the string data from a given filename
            Parse the string data into double

        Args:
            filename: the string which stores the txt file path and filename

        Returns:
            self.data: the float np array which stores all the read-in lines
            self.diff_time: the time difference between current time and REF_TIME
        """
        if not os.path.isfile(filename):
            print "Multimodal file %s does not exist." % filename
            return None

        feature_names = pd.read_excel('../annotation/column_names.xlsx', sheetname='descriptor')
        column_names = list(feature_names['Name'])

        df_mmdata = pd.read_csv(filename, sep=" ", header=None)
        df_mmdata.columns = column_names
        df_mmdata = df_mmdata.drop('extra', 1)
        realtime_obj = pd.to_datetime(df_mmdata['realtime_str'], format='%Y_%m_%d_%H_%M_%S_%f')
        df_mmdata['realtime_obj'] = realtime_obj
        df_mmdata.describe().to_csv('../temp/descirbe.csv')
        self.df_mmdata = df_mmdata
        self.diff_time = (realtime_obj-self.REF_TIME).dt.total_seconds().values

        print "%i rows read from %s " % (len(df_mmdata), filename)
        return 0


    def findClosetTimeIdx(self, anno_time):
        """
        find the row which is closest to anno_time
        return the index of found row in diff_time
        """
        annot_time_diff = (anno_time-self.REF_TIME).total_seconds()
        idx = np.argmin(np.abs(self.diff_time-annot_time_diff))
        return idx

    def locate_trial(self, df_trial, end):
        """
        given the starting and ending time for all the trials
        find out which trial does this request belong to
        this request is represented by @end, which is the ending time since
        beginning of the annotation
        we should just 1 and exactly 1 trial number
        """
        cand_trials = df_trial.loc[(df_trial['start'] < end) & (df_trial['end'] > end)]
        if len(cand_trials) == 0:
            print "!!! cannot locate corresponding trial for this request..."
            trial = -1
        elif len(cand_trials) == 1:
            trial = cand_trials.head(1)['trial'].values[0]
        else:
            print "!!! found more than one trials for this request..."
            trial = -1
        return trial

    def segment(self, anno_bank, subjID, time2idx, verbose):
        df_requests = anno_bank['df_requests']
        df_trial = anno_bank['df_trial']
        df_req = pd.DataFrame(columns=['subjID', 'trial', 'step', 'elapsed_time', \
                                    'instru_pre', 'instru_req', 'mm_data'])
        df_ope = pd.DataFrame(columns=['subjID', 'trial', 'step', 'elapsed_time', \
                                    'instru_pre', 'instru_req', 'mm_data'])
        prev_trial = -1
        trial_start_idx = -1

        for idx in df_requests.index:
            # ===============
            # work on requesting event
            # ===============
            # 2. trial, get current trial, get
            trial = self.locate_trial(df_trial, df_requests.loc[idx]['end'])
            if trial != prev_trial:
                prev_trial = trial
                trial_start_idx = idx
                continue

            # 4. elapsed_time
            elapsed_time = (df_requests.loc[idx]['start_time'] - \
                        df_requests.loc[idx-1]['end_time']).total_seconds()

            # 5. instru_pre
            instru_pre = df_requests.loc[idx-1]['name']

            # 6. instru_req
            instru_req = df_requests.loc[idx]['name']

            # 3. step (how many instruments have been used)
            step = idx - trial_start_idx

            # 1. subject ID
            subjID = subjID

            # 7. mm_data
            startIdx = time2idx[df_requests.loc[idx]['start_time']]
            endIdx = time2idx[df_requests.loc[idx]['end_time']]
            assert (startIdx < endIdx)
            mm_data = self.df_mmdata.loc[startIdx:endIdx]

            # construct row for requesting segment and append to df
            req = {}
            req['elapsed_time'] = elapsed_time
            req['instru_pre'] = instru_pre
            req['instru_req'] = instru_req
            req['trial'] = trial
            req['step'] = step
            req['subjID'] = subjID
            req['mm_data'] = mm_data
            df_req = df_req.append(req, ignore_index=True)

            # ===============
            # work on operation event
            # only modify the different ones
            # ===============
            elapsed_time = 0

            startIdx = time2idx[df_requests.loc[idx-1]['end_time']]
            endIdx = time2idx[df_requests.loc[idx]['start_time']]
            if startIdx == endIdx:
                endIdx = startIdx + 1 # enforce at least 1 data point
            assert (startIdx < endIdx)
            mm_data = self.df_mmdata.loc[startIdx:endIdx]

            # construct row for operating segment and append to df
            ope = {}
            ope['elapsed_time'] = elapsed_time
            ope['instru_pre'] = instru_pre
            ope['instru_req'] = instru_req
            ope['trial'] = trial
            ope['step'] = step
            ope['subjID'] = subjID
            ope['mm_data'] = mm_data
            df_ope = df_ope.append(ope, ignore_index=True)

        if verbose:
            df_req.to_csv('../temp/df_req.csv', index=False)
            df_ope.to_csv('../temp/df_ope.csv', index=False)

        return df_req, df_ope

def load_files(name):
    # timestamp for the videos
    timestamp_path = '../data_raw/Webcam_video/'+name+'/'+name+'_raw_ptz_timestamp.txt'
    df_timestamp = pd.read_csv(timestamp_path, sep=' ', header=None);
    df_timestamp.columns = ['index', 'time_str']
    df_timestamp['time_obj'] = pd.to_datetime(df_timestamp['time_str'], \
                                    format='%Y_%m_%d_%H_%M_%S_%f')

    # annotation files - requests
    anno_requests_path = '../annotation/'+name+'/'+name+'_'+'requests.txt'
    df_requests = pd.read_csv(anno_requests_path, sep=" ")
    df_requests['start_framecount'] = np.round(df_requests['start'] * 30.0).astype(int)
    df_requests['end_framecount'] = np.round(df_requests['end'] * 30.0).astype(int)
    df_requests['start_time'] = df_timestamp['time_obj'].loc[df_requests['start_framecount']].reset_index(drop=True)
    df_requests['end_time'] = df_timestamp['time_obj'].loc[df_requests['end_framecount']].reset_index(drop=True)
    df_requests['duration'] = df_requests['end'] - df_requests['start']
    print df_requests.tail()

    # annotation files - trial
    anno_trial_path = '../annotation/'+name+'/'+name+'_'+'trial.txt'
    df_trial = pd.read_csv(anno_trial_path, sep=" ")
    df_trial['start_framecount'] = np.round(df_trial['start'] * 30.0).astype(int)
    df_trial['end_framecount'] = np.round(df_trial['end'] * 30.0).astype(int)
    df_trial['start_time'] = df_timestamp['time_obj'].loc[df_trial['start_framecount']].reset_index(drop=True)
    df_trial['end_time'] = df_timestamp['time_obj'].loc[df_trial['end_framecount']].reset_index(drop=True)
    df_trial['duration'] = df_trial['end'] - df_trial['start']

    anno_bank = {}
    anno_bank['df_requests'] = df_requests
    anno_bank['df_trial'] = df_trial
    return anno_bank

def subject1special():
    # init multimodal reading
    parseMM = ReadMultiModal()
    verbose = 1 # 0->nothing, 1->print summary, 2->print details and draw plot

    subjID = 1

    # === trial 1 ===
    fname = str(subjID) + '_1'
    print "working on file: %s:" % fname

    # load the annotations
    anno_bank = load_files(fname)

    # read multimodal data
    parseMM.read('../data_raw/Multimodal_log/'+fname+'.txt')

    # build dict for mapping time to index for faster query
    time2idx = dict()
    for idx in anno_bank['df_requests'].index:
        start_time = anno_bank['df_requests'].loc[idx]['start_time']
        time2idx[start_time] = parseMM.findClosetTimeIdx(start_time)
        end_time = anno_bank['df_requests'].loc[idx]['end_time']
        time2idx[end_time] = parseMM.findClosetTimeIdx(end_time)

    # segment request and operation data
    df_req, df_ope = parseMM.segment(anno_bank, subjID, time2idx, verbose)

    # === trial 2~5 ===
    fname = str(subjID) + '_2'
    print "working on file: %s:" % fname

    # load the annotations
    anno_bank = load_files(fname)

    # read multimodal data
    parseMM.read('../data_raw/Multimodal_log/'+fname+'.txt')

    # build dict for mapping time to index for faster query
    time2idx = dict()
    for idx in anno_bank['df_requests'].index:
        start_time = anno_bank['df_requests'].loc[idx]['start_time']
        time2idx[start_time] = parseMM.findClosetTimeIdx(start_time)
        end_time = anno_bank['df_requests'].loc[idx]['end_time']
        time2idx[end_time] = parseMM.findClosetTimeIdx(end_time)

    # segment request and operation data
    df_req_2, df_ope_2 = parseMM.segment(anno_bank, subjID, time2idx, verbose)

    # merge the two
    df_req = df_req.append(df_req_2, ignore_index=True)
    df_ope = df_ope.append(df_ope_2, ignore_index=True)

    df_req.to_pickle('../data_proc/segmented/'+str(subjID)+'_df_req.pkl')
    df_ope.to_pickle('../data_proc/segmented/'+str(subjID)+'_df_ope.pkl')
    df_req.to_csv('../data_proc/segmented/'+str(subjID)+'_df_req.csv', index=False)
    df_ope.to_csv('../data_proc/segmented/'+str(subjID)+'_df_ope.csv', index=False)
    print "df_req and df_ope written into ../data_proc/segmented/"

def regular_segment():
    # init multimodal reading
    parseMM = ReadMultiModal()
    verbose = 1 # 0->nothing, 1->print summary, 2->print details and draw plot

    # for each subject
    subjs = [2,3,5,6,7,8]
    for subjID in subjs:
        fname = str(subjID) + '_1'
        print "working on file: %s:" % fname

        # load the annotations
        anno_bank = load_files(fname)

        # read multimodal data
        parseMM.read('../data_raw/Multimodal_log/'+fname+'.txt')

        # build dict for mapping time to index for faster query
        time2idx = dict()
        for idx in anno_bank['df_requests'].index:
            start_time = anno_bank['df_requests'].loc[idx]['start_time']
            time2idx[start_time] = parseMM.findClosetTimeIdx(start_time)
            end_time = anno_bank['df_requests'].loc[idx]['end_time']
            time2idx[end_time] = parseMM.findClosetTimeIdx(end_time)

        # segment request and operation data
        df_req, df_ope = parseMM.segment(anno_bank, subjID, time2idx, verbose)

        df_req.to_pickle('../data_proc/segmented/'+str(subjID)+'_df_req.pkl')
        df_ope.to_pickle('../data_proc/segmented/'+str(subjID)+'_df_ope.pkl')
        df_req.to_csv('../data_proc/segmented/'+str(subjID)+'_df_req.csv', index=False)
        df_ope.to_csv('../data_proc/segmented/'+str(subjID)+'_df_ope.csv', index=False)
        print "df_req and df_ope written into ../data_proc/segmented/"

def merge_segments():
    # for each subject
    subjs = [1,2,3,5,6,7,8]
    df_req_all = pd.DataFrame()
    df_ope_all = pd.DataFrame()
    for subjID in subjs:
        df_req = pd.read_pickle('../data_proc/segmented/'+str(subjID)+'_df_req.pkl')
        df_ope = pd.read_pickle('../data_proc/segmented/'+str(subjID)+'_df_ope.pkl')
        df_req_all = df_req_all.append(df_req, ignore_index=True)
        df_ope_all = df_ope_all.append(df_ope, ignore_index=True)

    df_req_all.to_pickle('../data_proc/segmented/df_req_all.pkl')
    df_ope_all.to_pickle('../data_proc/segmented/df_ope_all.pkl')
    df_req_all.to_csv('../data_proc/segmented/df_req_all.csv', index=False)
    df_ope_all.to_csv('../data_proc/segmented/df_ope_all.csv', index=False)

def proc_myo_yaw():
    """
    Manually remove the noise due to the yaw drifting.
    Since there is a constant drifting, we can use the difference instead of the
    actual value. The difference makes more sense.
    Due to the jump between -PI and PI, I zero-out any values larger than 2.
    This is a manual inspection and looking at the data.
    Read the pkl, change it, then save back to the same pkl and csv files
    !!! modifies original files !!!
    """
    df_req_all = pd.read_pickle('../data_proc/segmented/df_req_all.pkl')
    df_ope_all = pd.read_pickle('../data_proc/segmented/df_ope_all.pkl')
    for i in df_req_all.index:
        Myo_yaw = df_req_all.loc[i]['mm_data']['Myo_yaw'].values
        Myo_yaw[1:] = Myo_yaw[1:] - Myo_yaw[:-1]
        Myo_yaw[0] = 0
        Myo_yaw[np.abs(Myo_yaw) >= 2] = 0
        df_req_all.loc[i]['mm_data']['Myo_yaw'] = Myo_yaw

    for i in df_ope_all.index:
        Myo_yaw = df_ope_all.loc[i]['mm_data']['Myo_yaw'].values
        Myo_yaw[1:] = Myo_yaw[1:] - Myo_yaw[:-1]
        Myo_yaw[0] = 0
        Myo_yaw[np.abs(Myo_yaw) >= 2] = 0
        df_ope_all.loc[i]['mm_data']['Myo_yaw'] = Myo_yaw

    df_req_all.to_pickle('../data_proc/segmented/df_req_all.pkl')
    df_ope_all.to_pickle('../data_proc/segmented/df_ope_all.pkl')
    df_req_all.to_csv('../data_proc/segmented/df_req_all.csv', index=False)
    df_ope_all.to_csv('../data_proc/segmented/df_ope_all.csv', index=False)

def main():
##    subject1special()
##    regular_segment()
##    merge_segments()
    proc_myo_yaw()

if __name__ == '__main__':
    main()
