"""
Decode a CAN-bus mf4 file and export kinematic measurements to csv or parquet.
Can process individual files or search directories for mf4 files.

Usage:

python decode_can.py [-h] [-d DBC_FILEPATH] [-l LOG_DIRECTORY_OR_FILEPATH] [-o OUTDIR] [-f FORMAT] [-m]

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
                                                        'be searched for .mf4 files. '))
    parser.add_argument('-a', '--append', action='store_true', help='If True, multiple .mf4 files will be '
                                                        'appended to the same .csv. in alphabetical order. Only '
                                                        'select if files are consecutive! Otherwise, an individual '
                                                        '.csv file will be created for each .mf4 file.')
    parser.add_argument('-o', '--outdir', type=str, default='input directory', help='The output directory')
    parser.add_argument('-f', '--format', choices=['.csv', '.parquet'], default = '.csv',
                        help=('Output format. Choose ".csv" for human-readible files and ".parquet" for'
                              'memory efficiency.'))
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

    # output directories
    if args.outdir == 'input directory':
        dirs_out = [os.path.dirname(f) for f in filepaths_logs]
    else:
        dirs_out = [str(args.outdir)] * len(filepaths_logs) 

    # output names
    filenames_out = []
    for f in filepaths_logs:
        if args.outdir == 'input directory' and not args.append:
            filename_out = os.path.splitext(os.path.basename(filepaths_logs[0]))[0]
            filename_out += args.format
        else:            
            n0 = os.path.normpath(os.path.splitext(filepaths_logs_rel[0])[0])
            n0 = n0.replace(os.sep, '-')
            n0 = n0 + args.format
            filename_out = n0
        filenames_out.append(filename_out)

    # decode can files
    df_list = []
    if not args.mute:
        print("Decoding ...")
    for f in filepaths_logs:
            print(f"   {f}")
            df_i = can.process_can_edge(
                    [f],
                    {"LIN": [(filepath_dbc, 0)], "CAN": [(filepath_dbc, 0)]})
            df_list.append(df_i)

    # append
    if args.append:
        # Warn
        if len(df_list) > 1 and not args.mute:
            print((f"WARNING: Concatenating multiple log files in alphabetical order."
                f" This only makes sense if the log files contain directly consecutive"
                f" logs without breaks! Consider converting individual files if unsure."))

        df_list = [pd.concat(df_list, ignore_index=True)]
        # out directory
        if args.outdir == 'input directory':
            dirs_out = [args.logs]
        else: 
            dirs_out = [dirs_out[0]]
        # out filename
        filenames_out = [f"{filenames_out[0][:-4]}_{filenames_out[1][:-4]}"]

    if not args.mute:
        print("   done!")

    # choose measurements and rename
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

    if not args.mute:
        print("Extract kinematics and write ...")

    for df, dir_out, fname_out in zip(df_list, dirs_out, filenames_out):
        
        # filepath for writing
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
            if not args.mute:
                print(f'   Created missing output directory {dir_out}')
        
        if not filename_out.endswith(args.format):
            filename_out += args.format

        filepath_out = os.path.join(dir_out, fname_out)
        if not args.mute:
            print(f'   {filepath_out}')


        #extract kineamtics
        df = df[list(MEASUREMENTS_TO_EXTRACT.keys())]
        df = df.rename(MEASUREMENTS_TO_EXTRACT)

        # save
        if args.format == '.csv':
            df.to_csv(filepath_out, sep=';')
        else:
            df.to_parquet(filepath_out, engine="pyarrow", compression="snappy")
        
    if not args.mute:
        print("   done!")


if __name__ == "__main__":
    main()