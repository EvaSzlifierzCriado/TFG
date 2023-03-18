# Example matrices to try the code.
m4 = [[35, 40, 41, 45, 50], 
      [40, 40, 42, 46, 52], 
      [42, 46, 50, 55, 55], 
      [48, 52, 56, 58, 60], 
      [56, 60, 65, 70, 75]]
mConv = [[-2, -1, 0], 
         [-1, 1, 1], 
         [0, 1, 2]]


# Given a matrix M and a result matrix res, initializes
# first and last rows and columns of res with the 
# values of M.
def initializeEdges(M, res):
    res[0] = M[0]
    res[len(res)-1] = M[len(M)-1]
    for i in range(0, len(M)-1):
        res[i][0] = M[i][0]
    for i in range(0, len(M)-1):
        res[i][len(res)-1] = M[i][len(M)-1]
    return res

# Given a matrix M and a convolution matrix 3x3 conv, 
# returns the result of applying the convolution.
def matrixConv3x3(M, conv):
    cols = len(M[0])
    rows = len(M)
    res = [[0 for _ in range(cols)] for _ in range(rows)]
    initializeEdges(M, res)
    for i in range(1, len(M)-1):
        for j in range(1, len(M[0])-1):
            sumF1 = M[i-1][j-1] * conv[0][0] + M[i-1][j] * conv[0][1] + M[i-1][j+1] * conv[0][2]
            sumF2 = M[i][j-1] * conv[1][0] + M[i][j] * conv[1][1] + M[i][j+1] * conv[1][2]
            sumF3 = M[i+1][j-1] * conv[2][0] + M[i+1][j] * conv[2][1] + M[i+1][j+1] * conv[2][2]
            res[i][j] = sumF1 + sumF2 + sumF3
    return res
            


def main():
    print(matrixConv3x3(m4, mConv))
    
main()
