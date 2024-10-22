## How to run the scripts
You will need Python installed, as well as the following libraries:
- pip install pandas
- pip install scipy
- pip install matplotlib
- pip install tabulate

## NOTES ABOUT HOW TO USE SCRIPTS TO PROCESS FILES:

### Manually check/fix the datalogs
Some datalogs need to be manually fixed due to 2 different sampling intervals. When this issue is present, it is either some rows in the beginning or in the end of the file. It is possible to fix it by adjusting the timestamps' decimals so the interval between each row is the same. I.e., this is checked in the scripts.

### FCR-D Ramp UP
1) Run `synch_datalogs` script (e.g., `python synch_datalogs.py fcr_d_up_ramp/*.csv`).
This script will create a temp folder in the provided path, combine all datalogs with the same IP in the name and it will overwrite the timestamps to synch all datalogs (and corresponding metadata files) based on the first datalog of the directory.

2) Run `datalog_to_svk` script (e.g., `python datalog_to_svk.py -n CHOOSE_A_NAME fcr_d_up_ramp/temp_data/*.csv`).
This script will aggregate all units (each of which was previously combined in a single datalog). -n argument is a "system name" used as an identifier appended to output files' names. The script will also do some sanity checks and convert units in accordance to SVK specifications.

3) Run `fcr_data_split.py` (e.g., `python fcr_data_split.py -n CHOOSE_A_NAME FcrdUp_Cropped.csv FcrdUp_Cropped_metadata.csv`).
This script will output the test report and FcrdUp_RampTestHLHD files.

4) Run `svk_ramp_plot.py` (e.g., `python svk_ramp_plot.py FcrdUp_RampTestHLHD`).
It will plot the content of the mentioned file.

### FCR-D Ramp Down
1) Run `synch_datalogs` script with `-d` flag (e.g., `python synch_datalogs.py -d fcr_d_down_ramp/*.csv`).
This script will create a temp folder in the provided path, combine all datalogs with the same IP in the name and it will overwrite the timestamps to synch all datalogs (and corresponding metadata files) based on the first datalog of the directory.

2) Run `datalog_to_svk` script (e.g., `python datalog_to_svk.py -n CHOOSE_A_NAME fcr_d_down_ramp/temp_data/*.csv`).
This script will aggregate all units (each of which was previously combined in a single datalog). -n argument is a "system name" used as an identifier appended to output files' names. The script will also do some sanity checks and convert units in accordance to SVK specifications.

3) Run `fcr_data_split.py` (e.g., `python fcr_data_split.py -n CHOOSE_A_NAME FcrdDo_Cropped.csv FcrdDo_Cropped_metadata.csv`).
This script will output the test report and FcrdDo_RampTestHLHD files.

4) Run `svk_ramp_plot.py` (e.g., `python svk_ramp_plot.py FcrdDo_RampTestHLHD`).
It will plot the content of the mentioned file.
	
### FCR-D Sine
Same as in FCR-D Down (steps 1 to 2). It will also create the Nyquist plots.

### FCR-D LER tests:
1) Run `datalog_to_svk` script (e.g., `python datalog_to_svk.py -l -n CHOOSE_A_NAME ler/*.csv`).
This script will aggregate all units

2) Run `python svk_ler_plot.py --aem-fraction -n CHOOSE_A_NAME -o CHOOSE_OUTPUT_NAME FcrdDo_LERTestLLHD`

### Operational test (1h)
Same as in FCR-D Up (steps 1.2 to 1.4).


### Notes: 
* datalog_to_svk script has a regular expression that accepts input files with the name resulting of the synch_datalogs scripts (i.e., combined_datalog_IP). Update it so it accepts both formats (raw datalogs or combined ones).
