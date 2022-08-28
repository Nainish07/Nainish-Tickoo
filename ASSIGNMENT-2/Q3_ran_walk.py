#Question 3
import matplotlib.pyplot as plt
import Q1_random_num_gen as mr
def wolf_walk(n,x0):
    coordinate_x = [0]
    coordinate_y = [0]
    coordinate_z = [0]         # since our random generator only generates positive
    sum1 = 0                   # we multiply by -1^(some random number) to get negetive
    sum2 = 0                   # numbers as well
    sum3 = 0
    for i in range(0,n):       #sum the numbers to form coordinates
        sum1 = sum1 + (mr.My_Random(x0))*((-1)**round((mr.My_Random(x0/10000))))
        sum2 = sum2 + (mr.My_Random(2*x0+6))*((-1)**round((mr.My_Random(x0/100))))
        coordinate_x.append(sum2/32768)
        coordinate_y.append(sum1/32768)#to calculating the distance between
        x0 = mr.My_Random(x0)            # 2 consicutive coordinates
        z = (coordinate_x[i]-coordinate_x[i+1])**2 + (coordinate_y[i]-coordinate_y[i+1])**2
        #print(z)
        sum3 = sum3 + z
    rms = (sum3/n)**0.5
    #print(coordinate_x)
    #print(coordinate_y)
    print(rms,end=' is the net rms distance of the walk')
    print(" ") #for calculating the displacement just use the last coordinates
    d = ((coordinate_x[n])**2 + (coordinate_y[n])**2)**0.5  
    print(d,end=' is the net displacement of the walk')
    print(' ')
    #print(coordinate_x[n])
    #print(coordinate_y[n])
    plt.plot(coordinate_x,coordinate_y)
    plt.show()

with open("inputforq3.txt") as g:
    list1 = g.readlines()
for k in range(0, len(list1)):
    nums = list1[k].split()
    steps = int(nums[0])
    seed = int(nums[1])
    print("For ", steps, ":")
    wolf_walk(steps, seed)


"""Output

For  300 :
0.8038060080962061 is the net rms distance of the walk 
10.352843642578453 is the net displacement of the walk 

For  600 :
0.8208856013434496 is the net rms distance of the walk 
23.975577296279194 is the net displacement of the walk 

For  900 :
0.8101908214546021 is the net rms distance of the walk 
52.44702213717242 is the net displacement of the walk """

