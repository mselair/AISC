# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import tz

from Sleep.Data import parse_CyberPSG_Annotations_xml, standardize_CyberPSG_Annotations, standardize_timeAnnotations, standardize_annotationId, create_dayIndexes

def tile_annotations(dfAnnotations, dur_threshold):
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(dur_threshold, (int, float)):
        raise AssertionError(
            '[INPUT ERROR]: dur_threshold must be float or int format giving the maximum duration of a single annotation. All anotations above this duration threshold will be tiled.')

    if np.isnan(dur_threshold) or np.isinf(dur_threshold) or dur_threshold <= 0:
        raise AssertionError('[INPUT ERROR]: dur_threshold must be a valid number bigger than 0, not nan and not inf')

    dfAnnotations['tile'] = 0
    if (dfAnnotations['duration'] > dur_threshold).sum() > 0:
        annotation_idx = 0
        while annotation_idx < dfAnnotations.__len__():
            if dfAnnotations['duration'][annotation_idx] > dur_threshold:

                tdf = pd.DataFrame().append([dfAnnotations.iloc[annotation_idx]] * \
                                            int(np.ceil(dfAnnotations['duration'][annotation_idx] / dur_threshold)),
                                            ignore_index=True)

                tdelta = datetime.timedelta(0, dur_threshold)
                for idx, row in enumerate(tdf.iterrows()):
                    start_datetime = tdf.loc[idx, 'start'] + idx * tdelta
                    end_datetime = start_datetime + tdelta
                    if end_datetime > tdf.loc[idx, 'end']:
                        end_datetime = tdf.loc[idx, 'end']

                    tdf.loc[idx, 'start'] = start_datetime
                    tdf.loc[idx, 'end'] = end_datetime
                    tdf.loc[idx, 'duration'] = (end_datetime - start_datetime).seconds
                    tdf.loc[idx, 'tile'] = idx

                dfAnnotations = pd.DataFrame().append(
                    [
                        dfAnnotations.iloc[:annotation_idx],
                        tdf,
                        dfAnnotations.iloc[annotation_idx + 1:]
                    ], ignore_index=True
                )
            annotation_idx += 1
    return dfAnnotations

    '''
    if (dfAnnotations['duration'] > dur_threshold).sum() > 0:
        dfAnnotations['tile'] = 0
        annotation_idx = 0
        while annotation_idx < dfAnnotations.__len__():
            if dfAnnotations['duration'][annotation_idx] > dur_threshold:
                tdf = pd.DataFrame().append([dfAnnotations.iloc[annotation_idx]] * int(
                    np.floor(dfAnnotations['duration'][annotation_idx] / dur_threshold)),
                                            ignore_index=True)

                tdelta = datetime.timedelta(0, dur_threshold)
                for idx, row in enumerate(tdf.iterrows()):
                    tdf.loc[idx, 'start'] = tdf.loc[idx, 'start'] + idx * tdelta
                    tdf.loc[idx, 'end'] = tdf.loc[idx, 'end'] + idx * tdelta
                    tdf.loc[idx, 'duration'] = 30
                    tdf.loc[idx, 'tile'] = idx

                dfAnnotations = pd.DataFrame().append(
                    [
                        dfAnnotations.iloc[:annotation_idx],
                        tdf,
                        dfAnnotations.iloc[annotation_idx + 1:]
                    ], ignore_index=True
                )
            annotation_idx += 1

    return dfAnnotations
    '''

def filter_by_duration(dfAnnotations, duration):
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(duration, (int, float)):
        raise AssertionError(
            '[INPUT ERROR]: duration must be float or int format giving the maximum duration of a single annotation. All anotations above this duration threshold will be tiled.')

    if np.isnan(duration) or np.isinf(duration) or duration <= 0:
        raise AssertionError('[INPUT ERROR]: duration must be a valid number bigger than 0, not nan and not inf')

    dfAnnotations = dfAnnotations.loc[dfAnnotations['duration'] == duration].reset_index(drop=True)
    return dfAnnotations

def merge_neighbouring(df):
    new_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):
        appbl = True
        if idx > 0:
            if new_df.iloc[-1].annotation == row[1].annotation and new_df.iloc[-1].end == row[1].start:
                appbl = False

        if appbl == True:
            new_df = new_df.append(row[1], ignore_index=True)
        else:
            new_df.loc[new_df.__len__() - 1, 'end'] = row[1].end
        new_df.loc[new_df.__len__() - 1, 'duration'] = (new_df.loc[new_df.__len__() - 1, 'end'] - new_df.loc[new_df.__len__() - 1, 'start']).seconds
    return new_df

def filter_by_key(dfAnnotations, key, value):
    return dfAnnotations.loc[dfAnnotations[key] != value].reset_index(drop=True)

def get_time_by_key(df, key):
    if isinstance(key, list):
        value = 0
        for single_key in key:
            value += (df.duration[(df.annotation == single_key)]).sum()
        return value
    else:
        return (df.duration[(df.annotation == key)]).sum()

def get_fell_asleep_time(df, t_sleep_check=60, t_awake_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N1', 'N2', 'N3']):
    df = filter_by_key(df, 'annotation', 'Arrousal')
    # parameters
    t_sleep = datetime.timedelta(minutes=t_sleep_check) # interval since 1st asleep checked
    t_awake = datetime.timedelta(minutes=t_awake_threshold) # length of all awake cycles during the interval defined by t_sleep since the beginning of the sleep

    # get first asleep
    awake_to_sleep_changes = np.where(
        np.array([df.annotation[0] in sleep_cycle_tags] +
            [(df.annotation[k-1] in awake_tag) and (df.annotation[k] in sleep_cycle_tags) for k in range(1, df.__len__())
             ]))[0]

    fell_asleep_time = df.iloc[0].start
    for idx in awake_to_sleep_changes:
        start = df.start[idx]
        all_starts = df.start
        t_sleep_window_df = df.loc[(start <= all_starts) & (all_starts < (start + t_sleep))]
        time_awake = t_sleep_window_df.duration[t_sleep_window_df.annotation == awake_tag].sum()
        if time_awake < t_awake.seconds:
            fell_asleep_time = start
            break
    return fell_asleep_time

def get_awakening_time(df, t_awake_threshold=90, t_sleep_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N1', 'N2', 'N3']):
    df = filter_by_key(df, 'annotation', 'Arrousal')
    t_awake = datetime.timedelta(minutes=t_awake_threshold)
    t_sleep = datetime.timedelta(minutes=t_sleep_threshold)

    sleep_to_awake_changes = np.where(
        np.array([df.annotation[0] in sleep_cycle_tags] +
                 [(df.annotation[k-1] in sleep_cycle_tags) and (df.annotation[k] in awake_tag) for k in range(1, df.__len__())]
                 ))[0]

    if df.iloc[-1].annotation == awake_tag:
        last_awake_time = df.iloc[sleep_to_awake_changes[-1]].start
    else:
        last_awake_time = df.iloc[-1].end

    for idx in sleep_to_awake_changes:
        start = df.start[idx]
        all_starts = df.start
        t_awake_window_df = df.loc[(start <= all_starts) & (all_starts < (start + t_awake))]
        time_awake = t_awake_window_df.duration[t_awake_window_df.annotation == awake_tag].sum()
        time_asleep = get_time_by_key(t_awake_window_df, ['N1', 'N2', 'N3', 'REM'])
        if time_awake >= t_awake.seconds and time_asleep <= t_sleep_threshold:
            last_awake_time = start
            break
    return last_awake_time

def get_number_of_awakenings(df, awake_tag='AWAKE', n1_tag='N1', sleep_tags=['N2', 'N3', 'REM']):
    awake_bool = df.annotation == awake_tag
    n1_bool = df.annotation == n1_tag
    sleep_bool = np.zeros_like(n1_bool, dtype=bool)
    for tag in sleep_tags:
        sleep_bool = (sleep_bool) | (df.annotation == tag)

    awake_n1_bool = (awake_bool) | (n1_bool)

    n_awakenings = 0
    sleep_happened = False
    awake_happened = False
    for k in range(sleep_bool.shape[0]):
        if sleep_bool[k] == True:
            sleep_happened = True

        if awake_bool[k] == True:
            awake_happened = True
        else:
            if not awake_n1_bool[k] == True:
                awake_happened = False

        if sleep_happened == True and awake_happened == True:
            n_awakenings += 1
            sleep_happened = False
            awake_happened = False
    return n_awakenings

def get_number_of_sleep_stages(df, tags ='REM', delay=30):
    if isinstance(tags, str):
        tags = [tags]

    delay = datetime.timedelta(minutes=delay)
    bool_idxes = np.ones(df.__len__(), dtype=bool)
    for tag in tags:
        bool_idxes = (bool_idxes) & (df.annotation == tag)

    df = df.loc[bool_idxes].reset_index(drop=True)

    stage_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):
        if idx == 0:
            stage_df = stage_df.append(row[1], ignore_index=True)
        else:
            if (row[1].start - stage_df.iloc[-1].end).seconds >= delay.seconds:
                stage_df = stage_df.append(row[1], ignore_index=True)

    return (stage_df.annotation == tag).sum()

def is_sleep_complete(df, awake_tag='AWAKE'):
    return df.iloc[0].annotation == awake_tag == df.iloc[-1].annotation

def score_night(df, plot=False):
    df = filter_by_key(df, 'annotation', 'Arrousal')
    df = merge_neighbouring(df)

    fell_asleep_time = get_fell_asleep_time(df)
    awakening_time = get_awakening_time(df)
    sleep_complete = is_sleep_complete(df)

    sleep_df = df.loc[(df.start >= fell_asleep_time) & (df.start < awakening_time)].reset_index(drop=True)

    n_complete_sleep_cycles = get_number_of_sleep_stages(sleep_df, tags='REM', delay=30)
    n_awakenings = get_number_of_awakenings(sleep_df)

    n1_sleep_time = get_time_by_key(sleep_df, 'N1')
    n2_sleep_time = get_time_by_key(sleep_df, 'N2')
    n3_sleep_time = get_time_by_key(sleep_df, 'N3')
    rem_sleep_time = get_time_by_key(sleep_df, 'REM')
    awake_sleep_time = get_time_by_key(sleep_df, 'AWAKE')

    if plot == True: hypnogram(df)
    plt.stem([fell_asleep_time, awakening_time], [7, 7], linefmt='r', markerfmt='or', basefmt='r')

    return {
        'sleep_complete': sleep_complete,
        'fell_asleep_time': fell_asleep_time,
        'awakening_time': awakening_time,
        'n_complete_sleep_cycles': n_complete_sleep_cycles,
        'n_awakenings': n_awakenings,
        'n1_sleep_time': n1_sleep_time,
        'n2_sleep_time': n2_sleep_time,
        'n3_sleep_time': n3_sleep_time,
        'rem_sleep_time': rem_sleep_time,
        'awake_sleep_time': awake_sleep_time
    }

def hypnogram(orig_df):
    hypnogram_values = {
        'AWAKE': 6,
        'Arrousal': 5,
        'REM': 4,
        'N1': 3,
        'N2': 2,
        'N3': 1,
    }

    hypnogram_colors = {
        'AWAKE': '#e7b233',
        'Arrousal': '#d44b05',
        'REM': '#3500d3',
        'N1': '#2bc7c4',  # 2b7cc7
        'N2': '#2b5dc7',
        'N3': '#000000',
    }

    def set_hypnogram_properties(x, ref_dict):
        return ref_dict[x.annotation]

    orig_df['state_id'] = orig_df.apply(lambda x: set_hypnogram_properties(x, hypnogram_values), axis=1)
    orig_df['state_color'] = orig_df.apply(lambda x: set_hypnogram_properties(x, hypnogram_colors), axis=1)
    df_arrousals = orig_df.loc[orig_df.annotation == 'Arrousal'].reset_index(drop=True)
    df = orig_df.loc[orig_df.annotation != 'Arrousal'].reset_index(drop=True)
    new_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):
        appbl = True
        if idx > 0:
            if new_df.iloc[-1].state_id == row[1].state_id and new_df.iloc[-1].end == row[1].start:
                appbl = False

        if appbl == True:
            new_df = new_df.append(row[1], ignore_index=True)
        else:
            new_df.loc[new_df.__len__() - 1, 'end'] = row[1].end
    df = new_df

    x_start = np.array(df['start'])
    x_end = np.array(df['end'])
    for k, time_sample in enumerate(x_start): x_start[k] = time_sample.to_pydatetime()
    for k, time_sample in enumerate(x_end): x_end[k] = time_sample.to_pydatetime()

    plt.figure(dpi=200)
    plt.xlim(x_start[0], x_end[-1])
    # set background color for days
    for idx, day_id in enumerate(np.unique(df.day)):
        if idx % 2 == 0:
            background_color = 'gray'
            background_alpha = 0.1
        else:
            background_color = 'gray'
            background_alpha = 0.3

        day_start = x_start[df.day == day_id][0]
        day_end = x_end[df.day == day_id][-1]
        plt.axvspan(day_start, day_end, facecolor=background_color, alpha=background_alpha)

    # plot columns
    for idx, row in enumerate(df.iterrows()):
        val = row[1]['state_id']
        clr = row[1]['state_color']

        plt.fill_between(
            [x_start[idx], x_end[idx]],
            [val, val],
            color=clr,
            alpha=0.5,
            linewidth=0
        )

    for idx in range(df.__len__() - 1):
        val0 = df.state_id[idx]
        val1 = df.state_id[idx + 1]
        start0 = df.start[idx]
        start1 = df.start[idx + 1]
        end0 = df.end[idx]
        end1 = df.end[idx + 1]

        if val0 == val1:
            x = [start0, start1]
            y = [val0, val1]
        else:
            x = [start0, end0, start1]
            y = [val0, val0, val1]

        plt.plot(x, y, color='black', alpha=1, linewidth=1)

    x = [start1, end1]
    y = [val1, val1]
    plt.plot(x, y, color='black', alpha=1, linewidth=1)

    # plot arrousals
    for row in df_arrousals.iterrows():
        val = row[1].state_id
        clr = row[1].state_color
        plt.fill_between(
            # plt.plot(
            [row[1].start.to_pydatetime(), row[1].start.to_pydatetime(row[1].start), row[1].end.to_pydatetime(),
             row[1].end.to_pydatetime(row[1].end)],
            [0, val, val, 0],
            color=clr,
            alpha=1,
            linewidth=1
        )

    # format y ticks
    plt.yticks(list(hypnogram_values.values()), hypnogram_values.keys())
    for ticklabel in plt.gca().get_yticklabels():
        clr = hypnogram_colors[ticklabel._text]
        ticklabel.set_color(clr)

    # plot y grid
    for idx, key in enumerate(hypnogram_values.keys()):
        clr = hypnogram_colors[key]
        val = hypnogram_values[key]
        plt.plot([x_start[0], x_end[-1]], [val, val], color=clr, linewidth=0.7, alpha=0.7, linestyle=':')

    # format x_ticks
    plt.gcf().autofmt_xdate()
    formatter = mdates.DateFormatter("%H:%M", tz=tz.tzlocal())
    plt.gcf().get_axes()[0].xaxis.set_major_formatter(formatter)

    # plot hour x grid
    plt.grid(True, axis='x', alpha=1, linewidth=0.5, linestyle=':')

    # axes labels
    # plt.title('Days  ' + df.start[0].strftime('%d.%m') +'-' + df.iloc[-1].end.strftime('%d.%m'))
    plt.xlabel('\n Time [' + df.start[0].strftime('%d.%m.%Y') + ' - ' + df.iloc[-1].end.strftime('%d.%m.%Y') + ']')
    plt.ylabel('Sleep state')

def print_score(score):
    total_sleep_time =(score['awakening_time'] - score['fell_asleep_time']).seconds
    hours, remainder = divmod(total_sleep_time, 3600)
    minutes, seconds = divmod(remainder, 60)


    print('Sleep Complete: ', score['sleep_complete'])
    print('Falling asleep: ', score['fell_asleep_time'].strftime('%H:%M:%S'))
    print('Awakening: ', score['awakening_time'].strftime('%H:%M:%S'))
    print('Total Sleep Time: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
    print('Number of sleep cycles: ', score['n_complete_sleep_cycles'])
    print('Number of awakenings', score['n_awakenings'])
    print()
    print('Sleep-time awake')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['awake_sleep_time']),  score['awake_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N1')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n1_sleep_time']),  score['n1_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N2')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n2_sleep_time']),  score['n2_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N3')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n3_sleep_time']),  score['n3_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time REM')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['rem_sleep_time']),  score['rem_sleep_time']/total_sleep_time))



class SleepAnnotationsRef:
    annotationIdKey = 'annotationTypeId'
    time_format = '%Y-%m-%dT%H:%M:%S.%f'
    startTime_key = 'startTimeUtc'
    endTime_key = 'endTimeUtc'
    hour_cut = 16
    min_cut = 00

    @property
    def categories(self):
        return np.unique(self.dfAnnot.annotation)

    @property
    def counts(self):
        return np.array([(self.dfAnnot.annotation == cat).sum() for cat in self.categories])

    @property
    def counts_train(self):
        return np.array([(self.train.annotation == cat).sum() for cat in self.categories])

    @property
    def counts_validation(self):
        return np.array([(self.validation.annotation == cat).sum() for cat in self.categories])

    @property
    def crossval_idx(self):
        return self._crossvalidation_idx

    @crossval_idx.setter
    def crossval_idx(self, value):
        if self._max_crossval > value >= 0 and isinstance(value, int):
            self._crossvalidation_idx = value
        else:
            raise AssertionError('[VALUE ERROR]: crossvalidation_idx must be int >= 0 & <= ', self._max_crossval)

    @property
    def day(self):
        return int(self._max_day)

    @day.setter
    def day(self, value):
        if value <= self._max_day and isinstance(value, int):
            self._day = value
        else:
            raise AssertionError('[VALUE ERROR]: crossvalidation_idx must be int 0 & <= ', self._max_crossval,
                                 '. -1 indicates selection of all days.')

    @property
    def days(self):
        return int(self._max_day + 1)

    @property
    def validation(self):
        if self.day < 0:
            bool_vector = (self.dfAnnot.pick == self.crossval_idx)
        else:
            bool_vector = (self.dfAnnot.pick == self.crossval_idx) & (self.dfAnnot.day == self.day)

        return self.dfAnnot.loc[bool_vector].reset_index(drop=True)

    @property
    def train(self):
        if self.day < 0:
            bool_vector = (self.dfAnnot.pick != self.crossval_idx)
        else:
            bool_vector = (self.dfAnnot.pick != self.crossval_idx) & (self.dfAnnot.day == self.day)

        return self.dfAnnot.loc[bool_vector].reset_index(drop=True)

    def get_day(self, item):
        if item < 0:
            return self.dfAnnot
        else:
            return self.dfAnnot.loc[self.dfAnnot.day == item].reset_index(drop=True)

class SleepAnnotationsTraining(SleepAnnotationsRef):
    def __init__(self, path_xmlCyberPSG, crossval_groups=4):
        self._crossvalidation_idx = 0
        self._day = -1
        self._max_crossval = crossval_groups

        dfAnnot, AnnotationTypes = parse_CyberPSG_Annotations_xml(path_xmlCyberPSG)
        dfAnnot = standardize_annotationId(dfAnnot, self.annotationIdKey, AnnotationTypes)
        dfAnnot = standardize_timeAnnotations(dfAnnot, self.startTime_key, self.endTime_key, self.time_format)
        dfAnnot = create_dayIndexes(dfAnnot, hour=self.hour_cut, minute=self.min_cut)

        dfAnnot = tile_annotations(dfAnnot, 30)
        dfAnnot = filter_by_duration(dfAnnot, 30)
        self.dfAnnot = dfAnnot

        self._max_day = self.dfAnnot.day.max()

        self.dfAnnot['pick'] = 0

        categories = self.categories
        counters = np.zeros_like(self.counts)
        for idx, row in enumerate(self.dfAnnot.iterrows()):
            pos = np.where(categories == row[1].annotation)[0]
            self.dfAnnot.loc[idx, 'pick'] = counters[pos]
            counters[pos] += 1
            if counters[pos] > crossval_groups:
                counters[pos] = 0

class SleepAnnotationsScoring(SleepAnnotationsRef):
    def __init__(self, path_xmlCyberPSG, crossval_groups=4):
        self._crossvalidation_idx = 0
        self._day = -1
        self._max_crossval = crossval_groups

        dfAnnot, AnnotationTypes = parse_CyberPSG_Annotations_xml(path_xmlCyberPSG)
        dfAnnot = standardize_annotationId(dfAnnot, self.annotationIdKey, AnnotationTypes)
        dfAnnot = standardize_timeAnnotations(dfAnnot, self.startTime_key, self.endTime_key,
                                              self.time_format)
        dfAnnot = create_dayIndexes(dfAnnot, hour=self.hour_cut, minute=self.min_cut)
        dfAnnot = merge_neighbouring(dfAnnot)

        self.dfAnnot = dfAnnot
        self._max_day = self.dfAnnot.day.max()

