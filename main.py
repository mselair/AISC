import numpy as np
import pandas as pd
import datetime
#from CyberPSG_XML import myXML_AnnotationData
from lib.XML_parsing import parser_xml_CyberPSG
import xml.etree.ElementTree as ET
from collections import namedtuple
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import tz

DELIMITER = '\\'


def parse_CyberPSGAnnotationXML(path):
    Annotations = parser_xml_CyberPSG(path)

    annotationTypes = {}
    for annotType in Annotations['AnnotationTypes']['AnnotationType']:
        annotationTypes[annotType['id'].param] = annotType['name'].param

    dfAnnotations = pd.DataFrame()
    for annot in Annotations['Annotations']['Annotation']:
        temp = {
            'id': annot['id'].param,
            'modified': annot['created'].param,
            'startTimeUtc': annot['startTimeUtc'].param,
            'endTimeUtc': annot['endTimeUtc'].param,
            'annotationTypeId': annot['annotationTypeId'].param
        }
        dfAnnotations = dfAnnotations.append(temp, ignore_index=True)

    return dfAnnotations, annotationTypes


def standardize_annotationId(dfAnnotations, annotationIdKey, annotationIdDict=None):
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise ValueError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(annotationIdKey, str):
        raise ValueError('[INPUT ERROR]: Variable annotationIdKey must be of a type str.')

    if not annotationIdKey in dfAnnotations.keys():
        raise KeyError(
            '[KEY ERROR]: Key \'' + annotationIdKey + '\' is not present in the keys of pasted annotations: ' + dfAnnotations.keys())

    def translate_annotation(x):
        try:
            return annotationIdDict[x[annotationIdKey]]
        except:
            raise KeyError('[KEY ERROR]: No match found for an existing key in annotated data ' + x[
                annotationIdKey] + ' was not find upon the keys in dictionary of Annotation IDs: ' + annotationIdDict.keys())

    def copy_annotation(x):
        return x[annotationIdKey]

    if isinstance(annotationIdDict, dict):
        dfAnnotations['annotation'] = dfAnnotations.apply(lambda x: translate_annotation(x), axis=1)
    else:
        dfAnnotations['annotation'] = dfAnnotations.apply(lambda x: copy_annotation(x), axis=1)

    return dfAnnotations


def standardize_timeAnnotations(dfAnnotations, startTime_key, endTime_key, time_format):
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise ValueError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(startTime_key, str):
        raise ValueError('[INPUT ERROR]: Variable startTimeId_key must be of a type str.')

    if not isinstance(endTime_key, str):
        raise ValueError('[INPUT ERROR]: Variable endTimeId_key must be of a type str.')

    def convert_time(x, inp_key, time_format):
        if not inp_key in x.keys():
            raise KeyError(
                '[KEY ERROR]: Key \'' + inp_key + '\' is not present in the keys of pasted annotations: ' + x.keys())
        try:
            from_zone = tz.tzutc()
            to_zone = tz.tzlocal()
            # from_zone = tz.gettz('UTC')
            # to_zone = tz.gettz('America/Chicago')

            utc = datetime.datetime.strptime(x[inp_key], time_format)
            utc = utc.replace(tzinfo=from_zone)
            zone = utc.astimezone(to_zone)
            return zone
        except:
            raise AssertionError(
                '[DATETIME FORMAT ERROR]: Pasted time_format' + time_format + ' does not match time format of an annotation' +
                x[inp_key] + \
                'please see following web page for further information: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior')

    def get_duration(x):
        return (x['end'] - x['start']).seconds

    dfAnnotations['start'] = dfAnnotations.apply(lambda x: convert_time(x, startTime_key, time_format), axis=1)
    dfAnnotations['end'] = dfAnnotations.apply(lambda x: convert_time(x, endTime_key, time_format), axis=1)
    dfAnnotations['duration'] = dfAnnotations.apply(lambda x: get_duration(x), axis=1)
    return dfAnnotations


def create_dayIndexes(dfAnnotations, startTime_key='start', endTime_key='end', hour=12, minute=00):
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(hour, int):
        raise AssertionError('[INPUT ERROR]: hour variable must be of an integer type!')

    if not isinstance(minute, int):
        raise AssertionError('[INPUT ERROR]: minute variable must be of an integer type!')

    if not isinstance(startTime_key, str):
        raise AssertionError('[INPUT ERROR]: A startTime_key variable must be of a string type!')

    if not isinstance(endTime_key, str):
        raise AssertionError('[INPUT ERROR]: An endTime_key variable must be of a string type!')

    if hour < 0 or hour > 23:
        raise ValueError(
            '[VALUE ERROR] - An input variable hour_cut indicating at which hour days are separated from each other must be on the range between 0 - 23. Pasted value: ',
            hour)

    if minute < 0 or minute > 59:
        raise ValueError(
            '[VALUE ERROR] - An input variable min_cut indicating at which minute of an hour of a day the days are separated from each other must be on the range between 0 - 59. Pasted value: ',
            min)

    if not startTime_key in dfAnnotations.keys():
        raise KeyError('[KEY ERROR]: Key \'', startTime_key, '\' is not present in the keys of pasted annotations: ',
                       dfAnnotations.keys())

    if not endTime_key in dfAnnotations.keys():
        raise KeyError('[KEY ERROR]: Key \'', endTime_key, '\' is not present in the keys of pasted annotations: ',
                       dfAnnotations.keys())

    def check_datetime_format(x, key):
        if not isinstance(x[key], datetime.datetime):
            raise ValueError('[VALUE ERROR]: Annotation time \'', x[key], ' \' must be in the datetime format.')

    dfAnnotations.apply(lambda x: check_datetime_format(x, startTime_key), axis=1)
    dfAnnotations.apply(lambda x: check_datetime_format(x, endTime_key), axis=1)

    dfAnnotations['day'] = 0

    days, pos_indexes = np.unique([timestamp.day for timestamp in dfAnnotations[startTime_key]], return_index=True)
    months = [dfAnnotations[startTime_key][idx].month for idx in pos_indexes]
    years = [dfAnnotations[startTime_key][idx].year for idx in pos_indexes]
    day_thresholds = [datetime.datetime(years[idx], months[idx], day, hour, minute, 00, tzinfo=tz.tzlocal()) for
                      idx, day in
                      enumerate(days)]

    for idx in range(day_thresholds.__len__()):
        day_thr0 = day_thresholds[idx]
        if idx == day_thresholds.__len__() - 1:
            dfAnnotations.loc[(dfAnnotations[startTime_key] >= day_thr0), 'day'] = idx
        else:
            day_thr1 = day_thresholds[idx + 1]
            dfAnnotations.loc[
                (dfAnnotations[startTime_key] >= day_thr0) & (dfAnnotations[startTime_key] < day_thr1), 'day'] = idx

    return dfAnnotations


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


def filter_annotations_duration(dfAnnotations, duration):
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(duration, (int, float)):
        raise AssertionError(
            '[INPUT ERROR]: duration must be float or int format giving the maximum duration of a single annotation. All anotations above this duration threshold will be tiled.')

    if np.isnan(duration) or np.isinf(duration) or duration <= 0:
        raise AssertionError('[INPUT ERROR]: duration must be a valid number bigger than 0, not nan and not inf')

    dfAnnotations = dfAnnotations.loc[dfAnnotations['duration'] == duration].reset_index(drop=True)
    return dfAnnotations


def standardizeParsedAnnotationsPD(dfAnnotations,
                                   annotationIdKey=None,
                                   annotationIdDict=None,
                                   time_format='%Y-%m-%dT%H:%M:%S.%f',
                                   startTime_key='startTimeUtc',
                                   endTime_key='endTimeUtc',
                                   hour_cut=16,
                                   min_cut=00
                                   ):
    # dfAnnot = standardizeParsedAnnotationsPD(dfAnnot,
    #                                         annotationIdKey='annotationTypeId',
    #                                         annotationIdDict=AnnotationTypes,
    #                                         time_format='%Y-%m-%dT%H:%M:%S.%f',
    #                                         startTime_key='startTimeUtc',
    #                                         endTime_key='endTimeUtc',
    #                                         hour_cut=16,
    #                                         min_cut=00
    #                                         )

    dfAnnotations = standardize_annotationId(dfAnnotations, annotationIdKey, annotationIdDict)
    dfAnnotations = standardize_timeAnnotations(dfAnnotations, startTime_key, endTime_key, time_format)
    dfAnnotations = create_dayIndexes(dfAnnotations, hour=hour_cut, minute=min_cut)

    return dfAnnotations


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
        return self._day

    @day.setter
    def day(self, value):
        if value <= self._max_day and isinstance(value, int):
            self._day = value
        else:
            raise AssertionError('[VALUE ERROR]: crossvalidation_idx must be int 0 & <= ', self._max_crossval,
                                 '. -1 indicates selection of all days.')

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
    return new_df


class SleepAnnotationsTraining(SleepAnnotationsRef):
    def __init__(self, path_xmlCyberPSG, crossval_groups=4):
        self._crossvalidation_idx = 0
        self._day = -1
        self._max_crossval = crossval_groups

        dfAnnot, AnnotationTypes = parse_CyberPSGAnnotationXML(path_xmlCyberPSG)
        dfAnnot = standardize_annotationId(dfAnnot, self.annotationIdKey, AnnotationTypes)
        dfAnnot = standardize_timeAnnotations(dfAnnot, self.startTime_key, self.endTime_key, self.time_format)
        dfAnnot = create_dayIndexes(dfAnnot, hour=self.hour_cut, minute=self.min_cut)

        dfAnnot = tile_annotations(dfAnnot, 30)
        dfAnnot = filter_annotations_duration(dfAnnot, 30)
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

        dfAnnot, AnnotationTypes = parse_CyberPSGAnnotationXML(path_xmlCyberPSG)
        dfAnnot = standardize_annotationId(dfAnnot, self.annotationIdKey, AnnotationTypes)
        dfAnnot = standardize_timeAnnotations(dfAnnot, self.startTime_key, self.endTime_key,
                                              self.time_format)
        dfAnnot = create_dayIndexes(dfAnnot, hour=self.hour_cut, minute=self.min_cut)
        dfAnnot = merge_neighbouring(dfAnnot)

        self.dfAnnot = dfAnnot
        self._max_day = self.dfAnnot.day.max()


path = r'D:\MayoWork\MSEL\ExampleData\CyberPSGAnnotation.xml'
AnnotationsT = SleepAnnotationsTraining(path)
AnnotationsS = SleepAnnotationsScoring(path)

AWAKE = ['AWAKE']
REM = ['REM']
ARRAUSAL = ['Arrousal']
nREM = ['N1', 'N2', 'N3']

day1 = AnnotationsS.get_day(0)
day1 = merge_neighbouring(day1)


# beginning_sleep = n2_time = (day1.duration[(day1.annotation != 'N2')]).sum().loc[day1.]
# end_sleep = ''

def get_start_sleepA(df):
    df = df.loc[day1.annotation != 'Arrousal'].reset_index(drop=True)
    tthreshold_start = day1.start[0].replace(hour=20, minute=00, second=0)
    tthreshold_end = day1.iloc[-1].end.replace(hour=4, minute=00, second=0)
    tthreshold_awake = datetime.timedelta(minutes=10)
    tthreshold_sleep = datetime.timedelta(minutes=10)

    pos_idx = 0
    sleep_start = 0
    sleep_end = 0
    sleep_dur = 0
    awake_start = 0
    awake_end = 0
    awake_dur = 0
    awake = False
    sleep = False
    N3_bool = False
    while pos_idx < df.__len__():
        state = df.annotation[pos_idx]
        start = df.start[pos_idx]
        end = df.end[pos_idx]

        if state in (nREM + REM):
            if sleep is False:
                sleep_start = start
            sleep = True
            awake = False
            sleep_dur = end - sleep_start

        elif state in AWAKE:
            if sleep is True and awake is False:
                awake = True
                awake_start = start
            awake_dur = end - awake_start
            if awake_dur > tthreshold_awake and sleep_dur <= tthreshold_sleep:
                sleep = False
                awake = True

        pos_idx += 1

    return time_start


def get_start_sleep(df):
    def check_start_sleep_conditions(df, start, tthreshold_sleep, tthreshold_awake):
        df_hr = df.loc[(df.start >= start) & (df.start <= (start + tthreshold_sleep))]
        awake_dur = df_hr.loc[df_hr.annotation == 'AWAKE'].duration.sum()
        return awake_dur < tthreshold_awake.seconds

    df = df.loc[df.annotation != 'Arrousal'].reset_index(drop=True)
    tthreshold_start = day1.start[0].replace(hour=20, minute=00, second=0)
    tthreshold_end = day1.iloc[-1].end.replace(hour=4, minute=00, second=0)
    tthreshold_awake = datetime.timedelta(minutes=1)
    tthreshold_sleep = datetime.timedelta(minutes=60)

    start = df.start[df.annotation != 'AWAKE'].reset_index(drop=True)[0]
    while check_start_sleep_conditions(df, start, tthreshold_sleep, tthreshold_awake):
        start = df.start[df.annotation != 'AWAKE'].reset_index(drop=True)[0]

    # df_hr = df.loc[(df.start >= start) & (df.start <= start+tthreshold_sleep)]
    # awake_dur = df_hr.loc[df_hr.annotation == 'AWAKE'].duration.sum()
    # if awake_dur < tthreshold_awake.seconds:
    #    return start


time_start = get_start_sleep(day1)

# sleep time
n1_time = (day1.duration[(day1.annotation != 'N1')]).sum()
n2_time = (day1.duration[(day1.annotation != 'N2')]).sum()
n3_time = (day1.duration[(day1.annotation != 'N3')]).sum()
rem_time = (day1.duration[(day1.annotation != 'REM')]).sum()

sleep_time = n1_time + n2_time + n3_time + rem_time

# num of sleep cycles
sleep_cycles = 0
for k in range(day1.__len__() - 1):
    if day1.annotation[k] in nREM and day1.annotation[k + 1] in REM:
        sleep_cycles += 1

# rem latency
pos_rem1 = day1.index[day1.annotation == REM[0]][0]

temp = day1.iloc[:pos_rem1]
pos_awake = temp.index[temp.annotation == AWAKE[0]][-1]
rem_latency = day1.start[pos_rem1] - day1.end[pos_awake]

#3
#n1_dur =

# hypnogram(AnnotationsT.get_day(0))
# hypnogram(AnnotationsS.get_day(0))

hypnogram(AnnotationsS.get_day(0))

