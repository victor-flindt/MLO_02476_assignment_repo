from cmath import pi
import math

def equalidian(v1,v2):
    temp = 0
    answ = 0 
    for i in range(len(v1)):
        temp =  ((v1[i]-v2[i])**2)+temp
    answ = math.sqrt(temp)
    print(answ)
    answ = 0
    _sum = (sum(v1)-sum(v2))**2
    answ = math.sqrt(_sum)
    print(answ)
    return 0


a = [pi,1,-100]
b = [1,2,100]
equal = 0
print(equalidian(a,b)==equal)

