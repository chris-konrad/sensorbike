# Sensorbike: Postprocessing data captured from the instrumented Gazelle/Bosch Balance Assist Bicycles

This repository contains software to load and postprcocess data captured from the instrumented Gazelle/Bosch Balance Assist Bicycles. It enables to:

- Decode Swiftnav binary files and apply RTK corrections to GNSS data. (via RTKLib).

- Decode CAN log of the the bicycle OnBoard Unit (OBU).

- Load CAN and GNSS data into a Python environment.

- Time-synchronize CAN and GNSS data including drift compensation for long measurements.

- Transform measurements to common reference systems.

- Apply an Unscented Kalman Filter to the trajectory data to ensure consistency of GNSS and OBU data. 

> [!tip]
> The loaded data is made available as a `Track` object from the `trajdatamanager` toolbox. [This toolbox] enables easy highlevel trajectory operations like cropping, spatial filltering, subsampling, and plotting while supporting full `datetime.datetime` timestamps. However, it is still under development and doesn't have great documentation. If you prefer not to use the toolbox, get the data from the Track object as a dict `Track.to_dict()` or access the data array directly at `Track.data`. To retrieve individual features, Track objects can be accessed like a dictionary `Track['feature_name']`.

## Installation

Install the package and it's dependencies. Refer to `pyproject.toml` for an overview of the dependencies. 

1. Install `trajdatamanager`. See the its repository for installation instructions.

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

Their onboard unit already comes with:

- [VR IMU BN0086MEMs rate gyroscope](https://www.sparkfun.com/sparkfun-vr-imu-breakout-bno086-qwiic.html) mounted in the control box on the rack of the bikes:
  - Roll, Pitch, Yaw
  - Rollrate, Pitchrate, Yawrate
  - Linear accelerations
- Steer encoder:
  - Steer
  - Steerrate
- High Res BOSCH ABS speed sensor
  - wheelspeed

Additionally, a [SWIFTNAV Piksi Multi RTK GNSS]([Swift Navigation Support](https://support.swiftnav.com/support/solutions/articles/44001850752-piksi-multi-getting-started-guide)) can be mounted on the rack of the bikes. In conjunction with correction data from the fixed receiver of the [Dutch Permanent GNSS Array (DPGA)]([Dutch Permanent GNSS Array (DPGA)](https://gnss1.tudelft.nl/dpga/)) mounted on the EWI-tower, this adds high-precision localization to the bicycles. 

### Data Aquisition and Archiving

This toolbox assumes that you have captured data with the GNSS mounted on the rack of the bicycle and have logged the CAN bus of the bicycle. 

> [!warning]
> Currently, only logs captured with the CAN EDGE 2 are supported. The logs of the CAN2000 differ in format and can't be decoded with this toolbox.

Store the raw, encoded data (i.e. the `.M4F` files from the CAN logger and the `.sbp` files from the GNSS) in a file system with the following structure:

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

Note, that you also require the `.dbc` CAN bus definition to enable decoding the CAN data and GNSS correction data as well as an RTKLib configuration file for apply RTK corrections to the GNSS data. 

### Applying RTK-GNSS Corrections

Corrections to the GNSS data must be applied separately and externally to this toolbox. Refer to [swiftnav-processing](https://github.com/chris-konrad/swiftnav_processing) for instructions and the necessary software. 

## Using this toolbox

The most basic usage is demonstrated by `example.py` in the example toolbox. This includes a example `yaml` file for coniguration. More instructions and examples will follow ...

## Authors

- Christoph M. Konrad, c.m.schmidt@tudelft.nl
- Anna Marbus [Partgit  of this toolbox (CAN decoding) was taken from [bicycle-loc-and-state](https://gitlab.tudelft.nl/bicyclelab/bicycle-loc-and-state), developed during her research project.]

## License

The correction data was originally openly published by [TU Delft](https://gnss1.tudelft.nl/dpga/) without a license and is included here for convenience.
