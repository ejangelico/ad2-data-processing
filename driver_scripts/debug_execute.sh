#!/bin/bash

cd /g/g15/angelico/
source /g/g15/angelico/my_personal_env/bin/activate
python ad2-data-processing/reduction/preprocess_struck_5-11-23.py /p/lustre2/nexouser/data/StanfordData/angelico/hv-test-chamber/Run5/ds20/struck ad2-data-processing/ParseStruck/channel_map_template.txt
