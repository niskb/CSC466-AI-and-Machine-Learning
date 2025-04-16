import sys
def main():

    x0 = [1,2,3,4,5,6,7,8,9,10]
    x1 = [2.2,4.5,5.6,8.6,10.15,12.44,14.23,16.2,18.4,20.4]
    y = [0,1,0,1,0,1,0,0,1,1] # expected outputs of the network (do not confuse it

    w0 = 0.1
    w1 = 0.4
    w2 = -0.2
    w3 = 0.7
    w4 = 0.5
    w5 =0.5

    b0 = 0.22
    b1 = -0.1
    b2 = 0.3

    for i in range(0,100000):
        loss = 0
        for j in range(0,len(y)):
            s0 = x0[j] * w0 + x1[j] * w1 + b0
            s1 = x0[j] * w2 + x1[j] * w3 + b1
            a0 = s0
            a1 = s1
            s2 = a0 * w4 + a1 * w5 + b2
            a2 = s2
            
            loss += 0.5 * (y[j] - a2) ** 2

            dw4 = -(y[j] - a2) * a0
            dw5 = -(y[j] - a2) * a1
            db2  = -(y[j] - a2)
            dw0 = -(y[j] - a2) * w4 * x0[j]
            dw1 = -(y[j] - a2) * w4 * x1[j]
            dw2 = -(y[j] - a2) * w5 * x0[j]
            dw3 = -(y[j] - a2) * w5 * x1[j]
            db0  = -(y[j] - w4)
            db1  = -(y[j] - w5)

            w0 = w0 - 0.0001 * dw0
            w1 = w1 - 0.0001 * dw1
            w2 = w2 - 0.0001 * dw2
            w3 = w3 - 0.0001 * dw3
            w4 = w4 - 0.0001 * dw4
            w5 = w5 - 0.0001 * dw5
            b0 = b0 - 0.0001 * db0
            b1 = b1 - 0.0001 * db1
            b2 = b2 - 0.0001 * db2
        print('loss = ', loss)
    
    x0 = 2.7
    # 2*x + 0.03
    # 2*2.7 + 0.3 = 5.7
    x1 = 6.0
    print(w0,w1,w2,w3,w4,w5,b0,b1,b2)
    sa0 = x0 * w0 + x0 * w1 + b0
    sa1 = x0 * w2 + x1 * w3 + b1
    aa0 = sa0
    aa1 = sa1
    sa2 = aa0 * w4 + aa1 * w5 + b2
    aa2 = sa2
    #output = x0*w0 + x1*w1 + x0*w2 +   + b0
    print('output for (',x0,',',x1,') = ',aa2)

if __name__ == "__main__":
    sys.exit(int(main() or 0))