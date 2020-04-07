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


from Sleep.Data import parse_CyberPSG_Annotations_xml, standardize_CyberPSG_Annotations
from Sleep.Analysis import SleepAnnotationsScoring, SleepAnnotationsTraining, merge_neighbouring, tile_annotations, hypnogram, score_night, print_score

path = r'D:\MayoWork\MSEL\ExampleData\CyberPSGAnnotation.xml'
Annotations = SleepAnnotationsScoring(path)
for k in range(Annotations.days):
    day = Annotations.get_day(k)
    score = score_night(day, plot=True)
    print('#############################################')
    print('Day {0}'.format(k))
    print_score(score)






'''
day1 = filter_by_key(day1, 'annotation', 'Arrousal')
day1 = merge_neighbouring(day1)

fell_asleep_time = get_fell_asleep_time(day1)
awakening_time = get_awakening_time(day1)
sleep_complete = is_sleep_complete(day1)


sleep_df = day1.loc[(day1.start>=fell_asleep_time) & (day1.start<awakening_time)].reset_index(drop=True)

n_complete_sleep_cycles = get_number_of_sleep_stages(sleep_df, tags='REM', delay=30)
n_awakenings = get_number_of_awakenings(sleep_df)

n1_sleep_time = get_time_by_key(sleep_df, 'N1')
n2_sleep_time = get_time_by_key(sleep_df, 'N2')
n3_sleep_time = get_time_by_key(sleep_df, 'N3')
rem_sleep_time = get_time_by_key(sleep_df, 'REM')
awake_sleep_time = get_time_by_key(sleep_df, 'AWAKE')
asleep_to_awake = (awakening_time - fell_asleep_time).seconds


hypnogram(day1)
hypnogram(sleep_df)

'''









'''
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


#time_start = get_start_sleep(day1)










# sleep time
#n1_time = (day1.duration[(day1.annotation != 'N1')]).sum()
#n2_time = (day1.duration[(day1.annotation != 'N2')]).sum()
#n3_time = (day1.duration[(day1.annotation != 'N3')]).sum()
#rem_time = (day1.duration[(day1.annotation != 'REM')]).sum()

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

'''


