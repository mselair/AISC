import zmq
import numpy as np
from cloud_tools._mefclient_connection_variables import *

"""
class MefClient:
    RESPONSE_WAIT = 30
    def __init__(self, ports=None, server_ip=None):
        if isinstance(ports, type(None)):
            ports = PORT_MEF
        if isinstance(server_ip, type(None)):
            server_ip = IP_MEF

        context = zmq.Context(1)
        self.client = context.socket(zmq.REQ)
        self.poll = zmq.Poller()
        self.ports = ports
        self.server_ip = server_ip

    def request_data(self, path, channel, passwd='', start=None, stop=None, sample=0):
        # sample arg must be 1 if data are read by sample
        client = self.client
        for p in self.ports:
            client.connect(f"tcp://{self.server_ip}:{p}")

        self.poll.register(client, zmq.POLLIN)
        try:
            client.send_json({'path': path, 'channel': channel, 'passwd': passwd, 'start': int(round(start*1e6)), 'stop': int(round(stop*1e6)), 'sample': sample})
            socks = dict(self.poll.poll(self.RESPONSE_WAIT * 1000))
            if socks.get(client) == zmq.POLLIN:
                md = client.recv_json(flags=0, )
                msg = client.recv(flags=0, copy=True, track=False)
                buf = memoryview(msg)
                res_numpy = np.frombuffer(buf, dtype=md['dtype'])
                return res_numpy.reshape(md['shape']), md['fsamp']
            else:
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                self.poll.unregister(client)
                return False, f'Server response time elapsed {self.RESPONSE_WAIT} s'

        except Exception as exc:
            exce = f'{exc}'
            return False, exce
"""

class MefClient:
    __version__ = '2.0.0'
    def __init__(self,ports=None, server_ip=None, response_wait=30):
        if isinstance(ports, type(None)):
            ports = PORT_MEF
        if isinstance(server_ip, type(None)):
            server_ip = IP_MEF


        context = zmq.Context(1)
        self.client = context.socket(zmq.REQ)
        self.poll = zmq.Poller()
        self.ports = ports
        self.server_ip = server_ip
        self.response_wait = response_wait

    def request_data(self, path, channel, passwd='', start=None, stop=None, sample=0):
        # sample arg must be 1 if data are read by sample
        client = self.client
        for p in self.ports:
            client.connect(f"tcp://{self.server_ip}:{p}")

        self.poll.register(client, zmq.POLLIN)
        try:
            client.send_json({'path': path, 'channel': channel, 'passwd': passwd, 'start': int(round(start*1e6)), 'stop': int(round(stop*1e6)), 'sample': sample,
                              'flag':'data'})
            socks = dict(self.poll.poll(self.response_wait * 1000))
            if socks.get(client) == zmq.POLLIN:
                md = client.recv_json(flags=0, )
                msg = client.recv(flags=0, copy=True, track=False)
                buf = memoryview(msg)
                if md['error'] == 1:
                    return False, md['error_message'], False
                else:
                    res_numpy = np.frombuffer(buf, dtype=md['dtype'])
                    return True, res_numpy.reshape(md['shape']), md['fsamp']
            else:
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                self.poll.unregister(client)
                # rebind
                context = zmq.Context(1)
                self.client = context.socket(zmq.REQ)
                return False, f'Server response time elapsed {self.response_wait} s', False

        except Exception as exc:
            exce = f'{exc}'
            return False, exce, False


    def request_data_bulk(self, df):
        data = []
        fs = []
        for row in df.iterrows():
            row = row[1]
            outp = self.request_data(row['path'], row['channel'], start=row['start'], stop=row['end'])
            if not outp[0]:
                raise AssertionError('[ERROR] Data haven\'t been read \n' + str(row) + ' \n Error Message: ' + str(outp))
            data += [outp[1]]
            fs += [outp[2]]
        return data, fs



    def request_metadata(self, path, passwd=''):
        client = self.client
        for p in self.ports:
            client.connect(f"tcp://{self.server_ip}:{p}")
        self.poll.register(client, zmq.POLLIN)
        try:
            client.send_json({'path': path,  'passwd': passwd, 'flag': 'basic_info'})
            socks = dict(self.poll.poll(self.response_wait * 1000))
            if socks.get(client) == zmq.POLLIN:
                md = client.recv_json(flags=0, )
                if md['error'] == 1:
                    return False, md['error_message']
                else:
                    return True, md['bi']
            else:
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                self.poll.unregister(client)
                # rebind
                context = zmq.Context(1)
                self.client = context.socket(zmq.REQ)
                return False, f'Server response time elapsed {self.response_wait} s'

        except Exception as exc:
            exce = f'{exc}'
            return False, exce
