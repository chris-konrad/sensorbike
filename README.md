# Sensorbike: Postprocessing data captured from the instrumented Gazelle/Bosch Balance Assist Bicycles

This repository contains software to load and postprcocess data captured from the instrumented Gazelle/Bosch Balance Assist Bicycles. It enables to conveniently execute the full processing pipline outlined below and provides an environment to excute individual steps of the 
pipeline only. 

- Decode Swiftnav binary files and apply RTK corrections to GNSS data. (via RTKLib).

- Decode CAN log of the the bicycle OnBoard Unit (OBU).

- Load CAN and GNSS data into a Python environment.

- Time-synchronize CAN and GNSS data including drift compensation for long measurements.

- Transform measurements to common reference systems.

- Apply an Unscented Kalman Filter to the trajectory data to ensure consistency of GNSS and OBU data. 

> [!tip]
> The loaded data is made available as a `Track` object from the [`trajdatamanager`](https://github.com/chris-konrad/trajdatamanager) toolbox. [This toolbox] enables easy highlevel trajectory operations like cropping, spatial filltering, subsampling, and plotting while supporting full `datetime.datetime` timestamps. However, it is still under development and doesn't have great documentation. If you prefer not to use the toolbox, get the data from the Track object as a dict `Track.to_dict()` or access the data array directly at `Track.data`. To retrieve individual features, Track objects can be accessed like a dictionary `Track['feature_name']`. Deconding CAN files to csv can be done without the [`trajdatamanager`](https://github.com/chris-konrad/trajdatamanager) dependency using only this packages `canbus` module.

## Installation

Install the package and it's dependencies. Refer to `pyproject.toml` for an overview of the dependencies. 

1. Install [`trajdatamanager`](https://github.com/chris-konrad/trajdatamanager). See the repository for installation instructions. This is only required if you use the `bikedata` module. If you only want to decode CAN logs, you may skip this step. 

2. Clone this repository.
   
   ```bash
   git clone 
   ```

3. Install `sensorbike`. 
   
   ```bash
   cd sensorbike
   pip install .
   ```

## Getting Data and Preprocessing

### Bicycle Sensors

Their onboard unit publishes measurements from the following sensors on the bikes CAN bus:

- [VR IMU BN0086MEMs rate gyroscope](https://www.sparkfun.com/sparkfun-vr-imu-breakout-bno086-qwiic.html) mounted in the control box on the rack of the bikes:
  - Roll, Pitch, Yaw
  - Rollrate, Pitchrate, Yawrate
  - Linear accelerations
- Steer encoder:
  - Steer
  - Steerrate
- High Res BOSCH ABS speed sensor
  - wheelspeed

Both CAN loggers in the lab inventory are supported the CSS Electronics [CANedge2](https://www.csselectronics.com/products/can-bus-data-logger-wifi-canedge2) and [CL2000](https://www.csselectronics.com/products/can-bus-logger-interface-cl2000).

Additionally, a [SWIFTNAV Piksi Multi RTK GNSS]([Swift Navigation Support](https://support.swiftnav.com/support/solutions/articles/44001850752-piksi-multi-getting-started-guide)) can be mounted on the rack of the bikes. In conjunction with correction data from the fixed receiver of the [Dutch Permanent GNSS Array (DPGA)]([Dutch Permanent GNSS Array (DPGA)](https://gnss1.tudelft.nl/dpga/)) mounted on the EWI-tower, this adds high-precision localization to the bicycles. 



### Data Aquisition and Archiving

This toolbox assumes that you have captured data with the SwiftNav Piksi Multi GNSS mounted on the rack of the bicycle and have logged the CAN bus of the bicycle.  Store the raw, encoded data (i.e. the `.M4F`/`.txt` files from the CAN logger and the `.sbp` files from the GNSS) in a file system with the following structure:

```
.
├── experiment1
├── ...
└── experimentN
    ├── bike_gnss
    |   ├── correction_data
    |   ├── report
    |   ├── solution
    |   ├── XXXX-XXXX0.sbp
    |   ├── ...
    |   └── XXXX-XXXXN.sbp
    └── can-logger
        ├── XXXXXXXX
        |   └── XXXXXXXX.MF4
        └── motorcan.dbc
```

Note, that you require the `.dbc` CAN bus definition to enable decoding the CAN data and GNSS correction data as well as an RTKLib configuration file for apply RTK corrections to the GNSS data. Find the CAN bus definition `motorcan.dbc` in the Balance Assist Bicycle directory of the lab drive. 

Additionally, the bicycle paramters of the Balance Assist Bicycles are required for good UKF filter results.
These are not included in this repository and must be obtained from bicycleparameter like shown
here https://bicycleparameters.readthedocs.io/stable/gallery/examples/plot_balanceassistv1.html.

### Applying RTK-GNSS Corrections

Corrections to the GNSS data must be applied separately and externally to this toolbox. Refer to [swiftnav-processing](https://github.com/chris-konrad/swiftnav_processing) for instructions and the necessary software. 

## Using this toolbox

You can either run the full pipeline or individual steps.

### Full data processing
The most basic usage is demonstrated by `example_full-pipeline.py` in the example toolbox. This includes a example `yaml` file for coniguration. More instructions and examples will follow ...

### Decoding CAN-files only
You can decode and import CAN logs into a python environment without using the full data processing pipeline. For this, you need the filepath to the CAN file and the `.dbc` database definition. `sensorbike.canbus.process_can()` automatically detects if the logs are created from the CAN Edge 2 logger (`.mf4`) or the CL2000 logger (`.txt`), extracts can messages and stores the IMU, wheelspeed, and steer encoder measurements in a pandas dataframe. 

```python
import sensorbike.canbus as can
df = can.process_can(
    [FILEPATH_TO_CAN_LOG1, ..., FILEPATH_TO_CAN_LOGN,],
    FILEPATH_TO_DBC
)
```

Additionally, this package includes the script `scripts/decode_can.py` that decodes CAN log files and exports them to csv or parquet. 
Use it as below and call `--help` for more info on the arguments. 
```
> python decode_can.py decode_can.py [-h] -d DBC -l LOGS [-a] [-nr] [-i] [-o OUTDIR] [-f {.csv,.parquet}] [-m]
```
It supports:
- automatically decoding from CAN Edge 2 and CL 2000
- recursive and non-recursive search for log files (`--nonrecursive`)
- ignoring files for which a decoded output alread exists (`--ignoreexisting`)
- export to `.parquet` for efficient data storage in a binary format and `.csv` for human-readble text files. 

## Authors

- Christoph M. Konrad, c.m.konrad@tudelft.nl
- Anna Marbus [Part of this toolbox (CAN decoding) was taken from [bicycle-loc-and-state](https://gitlab.tudelft.nl/bicyclelab/bicycle-loc-and-state), developed during her research project.]

## License

This package is licensed under the terms of the [MIT license](https://github.com/chrismo-konrad/sensorbike/blob/main/LICENSE).
