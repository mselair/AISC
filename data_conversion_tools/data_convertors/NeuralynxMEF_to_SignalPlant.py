# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Converting annotations and mef signals from Neuralynx to CyberPSG-XML file. See example in readme file for this AISC sub-package.
"""

import os
import re
import argparse
import numpy as np
import pyedflib
from tqdm import tqdm

from mef_tools import MefReader



def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Converting mef file and annotations converted from Neuralynx to SignalPlant')
    parser.add_argument(
        '--path_mef', dest='PATH_MEF', required=True,
        help='Path to mef file which will be converted to edf file')
    parser.add_argument(
        '--path_txt', dest='PATH_TXT', required=True,
        help='Path to *.txt file generated from Neuralynx annotations')
    parser.add_argument(
        '--path_to', dest='PATH_TO', required=True,
        help='Dir path where SignalPlant complient annotations and signals will be generated')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Check windows or linux and set separator
    if os.name == 'nt': DELIMITER = '\\' # Windows
    else: DELIMITER = '/' # posix


    #path_mef = '/mnt/eplab/Personal/Inni/13apr_cube/mef3.mefd'
    #path_txt = '/mnt/eplab/Personal/Inni/13apr_cube/mef3.mefd/events.txt'
    #path_edf = '/mnt/Helium/filip/DATA/signalplant.edf'
    #path_annot = '/mnt/Helium/filip/DATA/signalplant.sel'

    args = parse_args()
    path_mef = args.PATH_MEF
    path_txt = args.PATH_TXT
    path_edf = os.path.join(args.PATH_TO, args.PATH_MEF.split(DELIMITER)[-1].split('.')[0] + '.edf')
    path_annot = os.path.join(args.PATH_TO, args.PATH_MEF.split(DELIMITER)[-1].split('.')[0] + '.sel')

    # Read events.txt - in mef, generated by Dan from Neuralynx native format
    with open(path_txt, 'r') as fid:
        txt = fid.read()
    txt = [sample for sample in re.split(r'[,][\s]|[\n]', txt) if sample.__len__() > 0]
    stamp1 = int(txt[0])
    note1 = txt[1]

    mef = MefReader(path_mef)

    channels = []
    starts = []
    ends = []
    fsamp = []
    nsamp = []
    signals = []

    for chbi in mef.bi:
        channels.append(chbi['name'])
        starts.append(chbi['start_time'][0])
        ends.append(chbi['end_time'][0])
        fsamp.append(chbi['fsamp'])
        nsamp.append(chbi['nsamp'])
        signals.append(
            mef.get_data(chbi['name'], chbi['start_time'][0], chbi['end_time'][0])[0]
        )



    channel_info = []
    for idx in range(3):
        ch_dict = {'label': channels[idx], 'dimension': 'uV', 'sample_rate': fsamp[idx], 'physical_max': signals[idx].max(), 'physical_min': signals[idx].min(), 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
        channel_info.append(ch_dict)
        signals[idx] = signals[idx].astype(np.int16)

    with pyedflib.EdfWriter(path_edf, 3, file_type=pyedflib.FILETYPE_EDFPLUS) as efid:
        efid.setSignalHeaders(channel_info)
        efid.writeSamples(signals)



    with pyedflib.EdfReader(path_edf) as efid:
        channels_edf_file = efid.getSignalLabels()
        fs = efid.getSampleFrequencies()[0]
        length = efid.getNSamples()[0]
        for idx in tqdm(range(signals.shape[0])):
            sig = efid.readSignal(idx, 0, length)
            if (sig - signals[idx]).sum() != 0:
                print('MEF Signal Conversion was not successful. Saved dat')


    # annotation
    print('Writting annotations')
    with open(path_annot, 'w') as fid:
        sts = starts[0] / 1e6
        fs = fsamp[0]
        annot_idx = 0

        fid.write("%SignalPlant ver.:1.2.6.5\n")
        fid.write("%Selection export from file:\n")
        fid.write("%{0}\n".format(path_edf.split('/')[-1]))
        fid.write("%SAMPLING_FREQ [Hz]:5000\n")
        fid.write("%CHANNELS_VALIDITY-----------------------\n")
        for k in range(channels.__len__()):
            fid.write("%{0}	1\n".format(channels[idx]))

        fid.write("%----------------------------------------\n")
        fid.write("%Structure:\n")
        fid.write("%Index[-], Start[sample], End[sample], Group[-], Validity[-], Channel Index[-], Channel name[string], Info[string]\n")
        fid.write("%Divided by: ASCII char no. 9\n")
        fid.write("%DATA------------------------------------\n")

        for file_idx in range(0, txt.__len__() - 1, 2):
            timestamp = int(txt[file_idx]) / 1e6
            note = txt[file_idx+1]
            start_samp = int(np.round((timestamp - sts)*fs)) + 10

            line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t%{6}\t{7}\n".format(annot_idx, start_samp, start_samp+10, 0, 0, 1, channels[0], note)
            fid.write(line)

            annot_idx += 1
    print('Annotations converted successfuly')
