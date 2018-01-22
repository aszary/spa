#! /usr/bin/env python

# ignored ranges in Q-mode
f = open("ignored.txt")

rngs = []
for line in f.readlines():
    res = line.split()
    rngs.append([int(res[0]), int(res[1])])
f.close()

print rngs
f = open("rngs.zap", "w")

for r in rngs:
    for i in xrange(r[0], r[1]+1, 1):
        f.write("%d\n" % i)

f.close()




print "Bye"
