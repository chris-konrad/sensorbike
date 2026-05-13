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

Below are the basic steps to install `sensorbike` in it's own virtual environment. Adapt as necessary to using within your own project. 
If you are new to Python and wonder what a virtual environment is, have a look [here](https://realpython.com/ref/best-practices/virtual-environments/) and [here](https://realpython.com/ref/tools/conda/). You may get (mini)conda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

1. Clone this repository and enter the repo folder.
   ```bash
   git clone https://github.com/chris-konrad/sensorbike.git
   ```

2. Create a virtual environment and activate it.
    ```bash
    conda env create -f sensorbike/environment.yml
    conda activate sensorbike
    ```

3. Manually install `trajdatamanager` following the steps below. First, clone latest version of `trajdatamanager` from https://github.com/chris-konrad/trajdatamanager. You may skip this step if you only want to do CAN decoding. 
    ```bash
    git clone https://github.com/chris-konrad/trajdatamanager.git
    ```
    If you plan to use the development branch of `sensorbike`, you must also switch to the development branch of `trajdatamanager` using the commands below. If you plan to use the latest release (main branch), skip the commands below.
    ```bash
    cd sensorbike
    git checkout development
    cd ../trajdatamanager
    git checkout development
    cd ..
    ```
    Finally, install `trajdatamanager`
    ```
    pip install trajdatamanager/.
    ```

4. Finally, install `sensorbike`. Optional dependencies are required for decoding CAN messages. 
   If you do not plan to decode CAN messages, run:
   ```bash
   pip install .
   ```
   If you plan to decode CAN messages logged by the CAN Edge 2 logger, CAN CL2000 logger, or both use one of the following:
   ```bash
   pip install sensorbike/.[canedge2] 
   ```
   or
   ```bash
   pip install sensorbike/.[cl2000] 
   ```
   or (careful, no space!)
   ```bash
   pip install sensorbike/.[canedge2,cl2000] 
   ```

Now you are ready start! 

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

Corrections to the GNSS data must be applied separately and externally to this toolbox. Refer to [rtkprocessing](https://github.com/chris-konrad/rtkprocessing) for instructions and the necessary software. 

## Using this toolbox

This section explains how to use the full processing pipeline (import IMU and GNSS data + UKF state estimation) and how to perform stand-alone CAN log decoding. For standalone GNSS decoding and RTK-processing, visit [rtkprocessing](https://github.com/chris-konrad/rtkprocessing).

### Full data processing

#### Demo
The most basic usage is demonstrated by `demo/example_full-pipeline.py`. This includes a example `yaml` file for configuration. 
Use it as below and call `--help` for more info on the arguments. Example data can be downloaded from https://doi.org/10.4121/f881dd80-b9f5-4322-9fd5-192034c9717f
```
> example_full-pipeline.py [--help] --datadir /path/to/the/toplevel/directory/of/the/example/data 
```

#### Using sensorbike in your own scripts
A typical processing pipeline will consist of:

1. Create an `sensorbike.bikedata.InstrumentedBicycleData()` object with the preferences of your choice. See the docstring for help.
```python
from sensorbike.bikedata import InstrumentedBicycleData()

dir_base = "path/to/your/data/"
experiment_name = "exp1" # data subfolder to process 
trial_name = "any-name-you-desire"

bikedata = InstrumentedBicycleData(dir_base, experiment_name, trial_name)
```
2. Load raw data. This automatically performs time synchronization and returns a Track object with the raw sensor data in their own reference frame. 
```python
data_raw = bikedata.load_raw()
```

3. Apply Kalman filter to retrieve bicycel states.
```python
states_filtered = bikedata.get_bicyclestates_filtered()
```

Check out the keyword-arguments of `InstrumentedBicycleData` for an overview how to customize the tool. 

### Reference Frames

Several reference frames are used within this repository. The complete list is below. 
You can select if you prefer the bicycle states to be expressed in the E-frame or the N-frame
using the `desired_reference_frame` kwarg of `InstrumentedBicycleData` a `BicycleStates` Track object can be 
transformed from E to N or inverse using `BicycleStates.transform_reference()`.
    B : Attached to the Frame of the bicycle, with B.x pointing forward, 
        B.y pointing to the right and B.z pointing downwards such that 
        B.x is parallel to the ground and B.z is parallel to gravity when
        the bicycle is upright. 
    N : The local reference frame as commonly defined in bicycle dynamics.
        N.x and N.y are fixed to the road surface and N.z points downwards.
    E : The local reference frame as commonly defined in traffic simulation.
        E.x and E.y are fixed to the road surface and E.z points upwards. 
        E.x and N.x are parallel. 
    Sgnss : The reference frame of the GNSS velocity. Sgnss.x equals the
        direction of the GNSS velocity vector derived from x-y coordinates.
        Sgnss.x and Sgnss.y are parallel to the ground and Sgnss.z is parallel
        to E.z. This ignores velocity in E.z direction, coming from the height
        of the GNSS changing when the bicycle tilts.
    Simu : The reference frame of the IMU velocities. Simu equals B save for small 
        rotations eps_x, eps_y, epx_z to account for IMU misalignment. 


### Decoding CAN-files only
You can decode CAN logs without using the full data processing pipeline. Use CL program `decodecan` installed with this repository to decode and export to `.csv` or `.parquet`. Use `sensorbike.canbus.process_can()` to decode within your own scripts.

#### Decode and export CAN logs to file.
Additionally, this package includes the program `decodecan` that decodes CAN log files and exports them to csv or parquet. 
Use it as below and call `--help` for more info on the arguments. 
```
> decodecan [-h] -d DBC -l LOGS [-a] [-nr] [-i] [-o OUTDIR] [-f {.csv,.parquet}] [-m] [-k {kinematics, messages, all}]
```
It supports:
- automatically decoding from CAN Edge 2 and CL 2000
- recursive and non-recursive search for log files (`--nonrecursive`).
- ignoring files for which a decoded output alread exists (`--ignoreexisting`).
- appending multiple can log files to the same output (`--append`)
- export to `.parquet` for efficient data storage in a binary format and `.csv` for human-readble text files (`--files`).
- extracting online kinematic measurements, status messages, or both (`--keys`).
- copy the filesystem of the input directory to an output directory.
- creates a decoding log.

#### Decode within your own scripts
You can decode and import CAN logs into a python environment without using the full data processing pipeline. For this, you need the filepath to the CAN file and the `.dbc` database definition. `sensorbike.canbus.process_can()` automatically detects if the logs are created from the CAN Edge 2 logger (`.mf4`) or the CL2000 logger (`.txt`), extracts can messages and stores the IMU, wheelspeed, and steer encoder measurements in a pandas dataframe. 

```python
import sensorbike.canbus as can
df = can.process_can(
    [FILEPATH_TO_CAN_LOG1, ..., FILEPATH_TO_CAN_LOGN,],
    FILEPATH_TO_DBC
)
```


## Authors

- Christoph M. Konrad, c.m.konrad@tudelft.nl
- Anna Marbus [Part of this toolbox (CAN decoding) was taken from [bicycle-loc-and-state](https://gitlab.tudelft.nl/bicyclelab/bicycle-loc-and-state), developed during her research project.]

## License

This package is licensed under the terms of the [MIT license](https://github.com/chrismo-konrad/sensorbike/blob/main/LICENSE).

The bicycle parameters in `/src/sensorbike/params` are derived from data provided by Jason Moore in [moorepants/BicycleParameters](https://github.com/moorepants/BicycleParameters), licensed under the [BSD-2-clause](https://github.com/chris-konrad/sensorbike/blob/main/src/rcid/params/LICENSE.txt) license.
