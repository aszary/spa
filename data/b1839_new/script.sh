#! /bin/bash
pmod -ext zap1 -zapfile rngs.zap data.gg  # creates  data.zap1
pmod -ext zap2 -zapfile rngs.zap -remove data.gg  # creates  data.zap2
#pmod -debase data.zap1  # removing baseline not needed
pspec -w -2dfs -lrfs -onpulse '188 267'  data.zap1 # 2DFS and LRFS creates data.1.2dfs data.lrfs
pspecDetect -v data.zap1 # detects P_3 somehow
pspec -p3fold "12.17 24" -w -onpulse '202 227'  data.zap1  # P3-folded profile in file data.p3fold

