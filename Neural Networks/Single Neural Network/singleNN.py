import sys
def main():
    # 2x+0.03
    # 2x+0.3
    # x= 1 ,  y= 2.03
    # x=2  ,  y 4.03
    # x=3 , y = 6.03
    # x = 4 y = 8.03
    #x0 = [1  ,2   ,3   ,4   ,5    ,6    ,7    ,8    ,9    ,10]
    #x1 = [2.2,4.02,6.05,8.01,10.04,11.99,14.05,16.01,18.04,20.0]
    #y  = [1  ,0   ,1   ,0   ,1    ,0    ,1    ,0    ,1    ,0]
    x0 = [1,2,3,4,5,6,7,8,9,10]
    x1 = [2.2,4.5,5.6,8.6,10.15,12.44,14.23,16.2,18.4,20.4]
    y = [0,1,0,1,0,1,0,0,1,1] # expected outputs of the network (do not confuse it

    w0 = 0.1
    w1 = -0.23
    b = 0.22

    for i in range(0,100000):
        loss = 0
        for j in range(0,len(y)):
            a = w0*x0[j] + w1 * x1[j] +b
            loss += 0.5 * (y[j] - a) ** 2
            dw0 = -(y[j] - a) * x0[j]
            dw1 = -(y[j] - a) * x1[j]
            db  = -(y[j] - a)

            w0 = w0 - 0.0001 * dw0
            w1 = w1 - 0.0001 * dw1
            b = b - 0.0001 * db
        print('loss = ', loss)
    
    x0 = 2.7
    # 2*x + 0.03
    # 2*2.7 + 0.3 = 5.7
    x1 = 5.24
    output = x0*w0 + x1*w1  + b
    print('output for (',x0,',',x1,') = ',output)

if __name__ == "__main__":
    sys.exit(int(main() or 0))