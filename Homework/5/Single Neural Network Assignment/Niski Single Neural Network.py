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
    w1 = -0.23
    b = 0.22

    #-------- train the single neuron network-------
    # need 100000 epochs, with 0.0001 learning rate
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

    # -----test for unknown data, on the trained network----------
    x0 = 6.90   # x coord. of point
    x1 = 23.0   # y coord. of point
    output = x0*w0 + x1*w1  + b
    print('output for (',x0,',',x1,') = ',output)

    x0 = 4.16    # x coord. of point
    x1 = 14.78   # y coord. of point
    output = x0*w0 + x1*w1  + b
    print('output for (',x0,',',x1,') = ',output)

if __name__ == "__main__":
    sys.exit(int(main() or 0))