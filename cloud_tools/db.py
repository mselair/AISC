# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import os
import zmq
import pickle
import json
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from tqdm import tqdm
from sqlalchemy.pool import NullPool
from sshtunnel import SSHTunnelForwarder
from cloud_tools._db_connection_variables import *
from AISC.models import KDEBayesianModel
from AISC import __version__ as aisc_version


class DatabaseHandler:
    __version__ = '0.0.1'
    def __init__(self, sql_db_name, sql_host=None, sql_user=None, sql_pwd=None, sql_port=None, ssh_host=None, ssh_user=None, ssh_pwd=None, ssh_port=None):
        if isinstance(sql_host, type(None)):
            sql_host = IP_SQL
        if isinstance(sql_user, type(None)):
            sql_user = USER_SQL
        if isinstance(sql_pwd, type(None)):
            sql_pwd = PW_SQL
        if isinstance(sql_port, type(None)):
            sql_port = PORT_SQL

        if isinstance(ssh_host, type(None)):
            ssh_host = IP_SSH
        if isinstance(ssh_user, type(None)):
            ssh_user = USER_SSH
        if isinstance(ssh_pwd, type(None)):
            ssh_pwd = PW_SSH
        if isinstance(ssh_port, type(None)):
            ssh_port = PORT_SSH

        self._ssh_host = ssh_host
        self._ssh_user = ssh_user
        self._ssh_pwd = ssh_pwd
        self._ssh_port = ssh_port

        self._sql_host = sql_host
        self._sql_port = sql_port
        self._sql_user = sql_user
        self._sql_pwd = sql_pwd
        self._sql_db_name = sql_db_name

        self._sql_connection = None
        self._ssh_tunnel = None
        self._engine = None


        self.open()
        self._init_sql_engine()

    def _init_sql_engine(self):
        if self.check_ssh_connection():
            self._engine = sqla.create_engine(
                'mysql+pymysql://{}:{}@{}:{}/{}'.format(self._sql_user, self._sql_pwd, 'localhost', self._ssh_tunnel.local_bind_port, self._sql_db_name), poolclass=NullPool)
        else:
            self._engine = sqla.create_engine(
                'mysql+pymysql://{}:{}@{}:{}/{}'.format(self._sql_user, self._sql_pwd, self._sql_host, self._sql_port, self._sql_db_name), poolclass=NullPool)

    def _open_sql(self):
        self._sql_connection = self._engine.connect()

    def _close_sql(self):
        self._sql_connection.close()

    def check_sql_connection(self):
        self._open_sql()
        self._close_sql()
        return True

    def _open_ssh(self):
        self._ssh_tunnel = SSHTunnelForwarder(
            (self._ssh_host, int(self._ssh_port)),
            ssh_username=self._ssh_user,
            ssh_password=self._ssh_pwd,
            remote_bind_address=(self._sql_host, int(self._sql_port)))
        self._ssh_tunnel.start()

    def check_ssh_connection(self):
        if self._ssh_tunnel:
            return self._ssh_tunnel.is_active
        return False

    def _close_ssh(self):
        if self.check_ssh_connection():
            self._ssh_tunnel.close()

    def open(self):
        if self._ssh_host:
            self._open_ssh()
            self.check_ssh_connection()

        self._init_sql_engine()
        self.check_sql_connection()

    def close(self):
        self._close_sql()
        self._close_ssh()

    def check_connection(self):
        self.check_ssh_connection()
        return self.check_sql_connection()

    def __del__(self):
        self.close()

    @property
    def db_name(self):
        return self._sql_db_name

    @db_name.setter
    def db_name(self, name):
        self._sql_db_name = name
        self.check_connection()

class SessionFinder(DatabaseHandler):
    #TODO: Enable searching for signals between multiple sessions
    __version__ = '0.0.1'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_mef_session(self, patient_id, uutc_start, uutc_stop):
        if not isinstance(uutc_start, (int, float, np.int, np.float)):
            raise TypeError('uutc_start has to be of a number type - int or float. Data type ' + type(uutc_start) + ' found instead.')
        if not isinstance(uutc_stop, (int, float, np.int, np.float)):
            raise TypeError('uutc_stop has to be of a number type - int or float. Data type ' + type(uutc_stop) + ' found instead.')

        uutc_start = int(round(uutc_start*1e6))
        uutc_stop = int(round(uutc_stop*1e6))


        self._sql_connection = self._engine.connect()
        query = f"SELECT uutc_start, uutc_stop, session, fsamp, channels FROM {self._sql_db_name}.Sessions where id='{patient_id}' and uutc_start<='{uutc_start}' and uutc_stop>='{uutc_stop}' order by uutc_start desc"
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()
        return df_data.loc[0, 'session']

    def find_mef_session_bulk(self, patient_id, df):
        sessions = []
        self._sql_connection = self._engine.connect()

        for row in tqdm(list(df.iterrows())):
            row = row[1]
            if not isinstance(row['start'], (int, float, np.int, np.float)):
                raise TypeError('start column has to be of a number type - int or float. Data type ' + type(row['start']) + ' found instead.')
            if not isinstance(row['end'], (int, float, np.int, np.float)):
                raise TypeError('end column has to be of a number type - int or float. Data type ' + type(row['end']) + ' found instead.')

            uutc_start = int(round(row['start']*1e6))
            uutc_stop = int(round(row['end']*1e6))
            query = f"SELECT uutc_start, uutc_stop, session, fsamp, channels FROM {self._sql_db_name}.Sessions where id='{patient_id}' and uutc_start<='{uutc_start}' and uutc_stop>='{uutc_stop}' order by uutc_start desc"
            df_data = pd.read_sql(query, self._sql_connection)
            sessions += [df_data.loc[0, 'session']]
        self._sql_connection.close()
        return sessions

    @property
    def patient_ids(self):
        self._open_sql()
        query = f"SELECT DISTINCT id FROM {self._sql_db_name}.Sessions"
        unique_ids = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        return unique_ids['id'].to_list()

class SleepClassificationModelDBHandler (DatabaseHandler):
    """Can read and save models from and into DB"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, cls, patient_id, channel, stimulation=None, kappa_test=0, score_training=None, score_validation=None, score_test=None, note=None):
        self._sql_connection = self._engine.connect()
        df = pd.DataFrame(
            [{
                'patient_id': patient_id,
                'channel': channel,
                'clf_name': cls.__name__,
                'aisc_version': aisc_version,
                'kappa': kappa_test,
                'stimulation': stimulation,
                'score_training' : json.dumps(score_training),
                'score_validation': json.dumps(score_validation),
                'score_test': json.dumps(score_test),
                'note': note,
                'classifier': pickle.dumps(cls)
            }]
        )
        df.to_sql('classifier_sleep', con=self._sql_connection, if_exists='append', index=False)
        self._sql_connection.close()


    def load_model(self, patient_id, channel):
        self._sql_connection = self._engine.connect()
        query = f"SELECT * FROM {self._sql_db_name}.classifier_sleep where patient_id='{patient_id}' and channel='{channel}'"
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()
        cls_string = df_data['classifier'][0]
        print(df_data.keys())
        return pickle.loads(cls_string), df_data.drop(columns='classifier')


