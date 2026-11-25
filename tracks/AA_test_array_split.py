
import numpy as np
"""
firstarray = np.array([1,2,3,4,5,6,7,8,9,10,11,12,23,34,45,67])

listoarray = np.array_split(firstarray,3)

print(listoarray)


arro = np.array([])

for i in range(len(listoarray)):
    print(listoarray[i])
    arro = np.concatenate((arro,listoarray[i]))

print(arro)

io_index = np.array([])
for i in range(len(listoarray)):
    r_peak = xxxPeaksP(listoarray[i])
    # get back thing of indexes
    rps = np.zeros(len(listoarray[i]))
    # have length of list
    for j in r_peak[i]:
        rps[j] = 1
    # have rps w. 1 for found, 0 for not found
    io_index = np.concatenate((np.array(io_index), rps))

r_peaks_2 = []
for i in io_index:
    if io_index[i] == 1:
        r_peaks_2.append(i)

np.array(r_peaks_2)
"""


#xt = np.array([1,2,4,3,6,4])

#xy = xt[2:5]

#print(xy)


xt = np.array([3,4,3,4,6,56,6,787,8,4,])
xt = abs(xt - np.mean(xt))
print(xt)
ym = np.argmax(xt)
print(ym)


print(xt[3:3])