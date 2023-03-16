# Example matrices to try the code.
m1 = [[1, 2], [3, 4]]
m2 = [[5, 6], [7, 8]]
m3 = [[9, 10], [11, 12]]

# Given a list of matrixes M, returns the multiplication of all
# the matrices of the list using a loop. 
def matrixMult(M):
    result = M[0]
    for i in range(1, len(M)):
        matrix = M[i]
        if len(result[0]) != len(matrix): # C1 must be equal to F2
            raise ValueError('Matrices cannot be multiplicated')
        newResult = []
        for j in range(len(result)):
            row = []
            for k in range(len(matrix[0])):
                element = 0
                for l in range(len(matrix)):
                    element += result[j][l] * matrix[l][k]
                row.append(element)
            newResult.append(row)
        result = newResult
    return result

def main():
    print(matrixMult([m1, m2, m3]))
    
main()