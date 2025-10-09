"""
Decode a CAN-bus mf4 file and export kinematic measurements to csv.

Usage:

python can2csv.py [-h] [-d DBC_FILEPATH] [-l LOG_DIRECTORY_OR_FILEPATH] [-o OUTDIR] [-f OUTFILENAME] [-m]

@author: Christoph M. Konrad
"""

import os
import argparse
import sensorbike.canbus as can
import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(prog='can2csv',
                                     description='Decode one or multiple .mf4 CAN logs and export the kinematic measurements to .csv')
    parser.add_argument('-d', '--dbc', type=str, help='File path to the CAN bus definition file (.dbc file)')
    parser.add_argument('-l', '--logs', type=str, help=('File path of a single .mf4 CAN log file or a '
                                                        'directory which contains log files. If a directory is'
                                                        ' supplied. The directory and all subdirectories will '
                                                        'be searched for .mf4 files. Multiple files will be '
                                                        'appended to the same .csv. in alphabetical order. Only '
                                                        'supply a directory of log files if they are consecutive!'))
    parser.add_argument('-o', '--outdir', type=str, default='input directory', help='The output directory')
    parser.add_argument('-f', '--outfilename', type=str, default='input filename', help='The output filename')
    parser.add_argument('-m', '--mute', action='store_true', help='Mutes verbosity.')
    
    return parser.parse_args()
    

def main():

    args = parse_args()

    # set up file paths
    filepath_dbc = can.verify_filepath_dbc(args.dbc)

    filepaths_logs = []
    if os.path.isdir(args.logs):
        filepaths_logs_rel = can.list_canlogs(args.logs, verbose=not args.mute)
        filepaths_logs = [os.path.join(args.logs, f) for f in filepaths_logs_rel]
        if len(filepaths_logs)==0:
             raise FileNotFoundError(f"No CAN logs found in {args.logs}!")
    else:
        filepaths_logs = [can.verify_filepath_mf4(args.logs)]
        if not args.mute:
            print(f"Found 1 log file:")
            print(f"      1: {filepaths_logs[0]}")

    # decode can files
    df_list = []
    if not args.mute:
        print("Decoding ...")
    for f in filepaths_logs:
            df_i = can.process_can_edge(
                    filepaths_logs,
                    {"LIN": [(filepath_dbc, 0)], "CAN": [(filepath_dbc, 0)]})
            df_list.append(df_i)
    
    df = pd.concat(df_list, ignore_index=True)
    if not args.mute:
        print("    done!")

    # choose measurements and rename
    if not args.mute:
        print("Extracting kinematics ...")
    MEASUREMENTS_TO_EXTRACT = {
        'gyro_z': 'gyro_z_rad/s',
        'gyro_y': 'gyro_y_rad/s',
        'gyro_x': 'gyro_x_rad/s',
        'accel_z': 'accel_z_m/s2', 
        'accel_y': 'accel_y_m/s2',
        'accel_x': 'accel_x_m/s2',
        'yaw': 'yaw_rad',
        'pitch': 'pitch_rad',
        'roll': 'roll_rad',
        'ws_rear': 'wheelspeed_rear_rev/s',
        'LWS_ANGLE': 'steer_deg',
        'LWS_SPEED': 'steer_rate_deg/s',
    }
    df = df[list(MEASUREMENTS_TO_EXTRACT.keys())]
    df = df.rename(MEASUREMENTS_TO_EXTRACT)
    if not args.mute:
        print("    done!")


    # output directory and filename
    if args.outdir == 'input directory':
        if os.path.isdir(args.logs):
            dir_out = args.logs
            dir_out = os.path.dirname(args.logs)
    else:
        dir_out = os.path.dirname(args.outdir)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
        if not args.mute:
            print(f'Created missing output directory {dir_out}')

    if args.outfilename == 'input filename':
        if len(filepaths_logs)==1:
            filename_out = os.path.splitext(os.path.basename(filepaths_logs[0]))[0]
        else:            
            n0 = os.path.normpath(os.path.splitext(filepaths_logs_rel[0])[0])
            n0 = n0.replace(os.sep, '-')
            n1 = os.path.normpath(os.path.splitext(filepaths_logs_rel[-1])[0])
            n1 = n1.replace(os.sep, '-')
            filename_out = f"{n0}_{n1}"
        filename_out += ".csv"
    else:
        filename_out = args.outfilename
    
    if not filename_out.endswith('.csv'):
        filename_out += '.csv'

    filepath_out = os.path.join(dir_out, filename_out)

    # save
    if not args.mute:
        print(f'Writing measurements to {filepath_out}')
    df.to_csv(filepath_out, sep=';')
    if not args.mute:
        print("    done!")

    # Warn
    if not args.mute:
        if len(filepaths_logs)>1:
            print((f"WARNING: Concatenated multiple log files in alphabetical order."
                   f"This only makes sense if the log files contain directly consecutive"
                   f" logs without breaks! Consider converting individual files if unsure."))


if __name__ == "__main__":
    main()