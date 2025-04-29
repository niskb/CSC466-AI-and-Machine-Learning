import sys
def main():

    # y = 3 * x + 2.3
    # x0 corresponds to x coordinate, x1 corresponds to y coordinate of a point
    # to output a 0, if the point is above the line, it's output is to be 1

    #------create some training data---------
    # need 20 data points
    x0 = [   1,    2,     3,     4,     5,     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,    19,    20]
    x1 = [5.29, 8.31, 11.29, 14.31, 17.29, 20.31, 23.29, 26.31, 29.29, 32.31, 35.29, 38.31, 41.29, 44.31, 47.29, 50.31, 53.29, 56.31, 59.29, 62.31]
    y =  [   0,    1,     0,     1,     0,     1,     0,     1,     0,     1,     0,     1,     0,     1,     0,     1,     0,     1,     0,     1] # expected outputs of the network (do not confuse it with y coordinate of a point)

    #-------initialize weights and biases-----
    w0 = 0.1
    w1 = 0.4
    w2 = -0.2
    w3 = 0.7
    w4 = 0.5
    w5 =0.5

    b0 = 0.22
    b1 = -0.1
    b2 = 0.3

    #-------- train the multi layer neuron network-------
    # need 100000 epochs, with 0.0001 learning rate
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
    
    # -----test for unknown data, on the trained network----------
    x0 = 6.90   # x coord. of point
    x1 = 23.0   # y coord. of point
    print(w0,w1,w2,w3,w4,w5,b0,b1,b2)
    sa0 = x0 * w0 + x0 * w1 + b0
    sa1 = x0 * w2 + x1 * w3 + b1
    aa0 = sa0
    aa1 = sa1
    sa2 = aa0 * w4 + aa1 * w5 + b2
    aa2 = sa2
    #output = x0*w0 + x1*w1 + x0*w2 +   + b0
    print('output for (',x0,',',x1,') = ',aa2)

    x0 = 4.16    # x coord. of point
    x1 = 14.78   # y coord. of point
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
