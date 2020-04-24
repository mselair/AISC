# Copyright (C) 2020, Mayo Clinic - Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
import datetime
from AISC.XML_parsing import parser_xml_CyberPSG
from dateutil import tz

def parse_CyberPSG_Annotations_xml(path):
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


def standardize_CyberPSG_Annotations(dfAnnotations,
                                   annotationIdKey='annotationTypeId',
                                   annotationIdDict=None,
                                   time_format='%Y-%m-%dT%H:%M:%S.%f',
                                   startTime_key='startTimeUtc',
                                   endTime_key='endTimeUtc',
                                   hour_cut=16,
                                   min_cut=00
                                   ):
    dfAnnotations = standardize_annotationId(dfAnnotations, annotationIdKey, annotationIdDict)
    dfAnnotations = standardize_timeAnnotations(dfAnnotations, startTime_key, endTime_key, time_format)
    dfAnnotations = create_dayIndexes(dfAnnotations, hour=hour_cut, minute=min_cut)

    return dfAnnotations





