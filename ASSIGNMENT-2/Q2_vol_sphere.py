#Question 2 
import matplotlib.pyplot as plt
import Q1_random_num_gen as mr

def volume(n,x0):
    x = []
    y = []
    z = []
    ctr = 0
    for i in range(0,n):
        x.append(i/n) #for x coordinate
        y.append(mr.My_Random(x0)/32768) # for y coordinate
        z.append(mr.My_Random(mr.My_Random(x0))/32768) # for z coordinate
        x0 = mr.My_Random(x0)
        d = (x[i])**2 + y[i]**2 +z[i]**2 
        if d < 1: #check for distance to be less than 1
            ctr = ctr + 1
    return (ctr/n)
    
with open("inputforq2.txt") as g:
    numlist = g.readlines()
inp = numlist[0].split()
step =int(inp[0])
seed = int(inp[1])


vol = volume(step, seed)
print("Volume of the octant of the sphere =  ", vol)

"""OUTPUT

Volume of the octant of the sphere =   0.5236295053984208"""