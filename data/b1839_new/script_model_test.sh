#! /bin/bash
#pmod -debase single_pulses.ascii  # removing baseline not needed
#pspec -w -2dfs -lrfs -onpulse '1 198'  -header "p0 1.839944315159" -header "dt 0.009199721575749999" single_pulses.ascii  # 2DFS and LRFS creates data.1.2dfs data.lrfs
#pspecDetect -v single_pulses.ascii  # detects P_3 somehow
#pspec -p3fold "12.17 24" -w -onpulse '202 227'  data.zap1  # P3-folded profile in file data.p3fold
#pspec -p3fold "12.3 24" -w -onpulse '30 70' -header "p0 1.839944315159" -header "dt 0.009199721575749999" single_pulses.ascii
pspec -p3fold "12.3 24" -w -onpulse '30 70' -header "p0 1.839944315159" -header "dt 0.009199721575749999" single_pulses_test.ascii
#pspec -p3fold "12.3 24" -w -onpulse '30 70' -header "p0 1.839944315159" -header "dt 0.009199721575749999" single_pulses_noiseless.ascii
