[1, 0, 1, 1, 0, 0, 1, 0, 0, 1], 
[0, 1, 1, 0, 1, 0, 1, 0, 1, 1], 
[0, 0, 1, 0, 1, 0, 0, 1, 0, 0], 
[1, 0, 1, 0, 1, 1, 1, 1, 1, 1], 
[0, 1, 0, 1, 1, 0, 0, 0, 0, 1], 
[0, 0, 1, 0, 1, 1, 1, 0, 1, 0], 
[0, 1, 0, 1, 0, 1, 0, 0, 1, 1], 
[1, 0, 0, 0, 1, 1, 1, 1, 0, 1], 
[1, 1, 1, 1, 1, 1, 1, 0, 1, 0], 
[1, 1, 1, 1, 0, 1, 0, 0, 1, 1]]


from collections import deque

class Solution:
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        hs_alt = set()
        hs_pac = set()
        pac_mat = [[False]*len(matrix[0]) for _ in range(len(matrix))]
        atl_mat = [[False]*len(matrix[0]) for _ in range(len(matrix))]
        output = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if self.pacific(i, j, matrix, pac_mat, hs_pac) and self.atlantic(i, j, matrix, atl_mat, hs_alt):
                    output.append([i,j])
        return output
    
    def pacific(self, i, j, matrix, pac_mat, hs):
        if (i, j) in hs:
            return pac_mat[i][j]
        q = deque()
        q.append((i, j))
        
        while len(q) != 0:
            for _ in range(len(q)):
                loc_i, loc_j = q.popleft()
                for hbr_i, hbr_j in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                    if hbr_i < 0 or hbr_j < 0:
                        hs.add((i, j))
                        pac_mat[i][j] = True
                        return True
                    #elif hbr_i > len(matrix) - 1 or hbr_j > len(matrix[0]) - 1:
                    #    atl_mat[i][j] = True
                    elif hbr_i < len(matrix) and hbr_j < len(matrix[0]) and matrix[hbr_i][hbr_j] <= matrix[loc_i][loc_j]:
                        if (hbr_i, hbr_j) in hs and pac_mat[hbr_i][hbr_j] == True:
                            hs.add((i, j))
                            pac_mat[i][j] = True                            
                            return True
                        elif (hbr_i, hbr_j) not in hs:
                            q.append((hbr_i, hbr_j))
        hs.add((i, j))
        pac_mat[i][j] = False
        return False
    
    def atlantic(self, i, j, matrix, atl_mat, hs):
        if (i, j) in hs:
            return atl_mat[i][j]
        q = deque()
        q.append((i, j))
        
        while len(q) != 0:
            for _ in range(len(q)):
                loc_i, loc_j = q.popleft()
                for hbr_i, hbr_j in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                    if hbr_i > len(matrix) - 1 or hbr_j > len(matrix[0]) - 1:
                        hs.add((i, j))
                        atl_mat[i][j] = True
                        return True
                    #elif hbr_i > len(matrix) - 1 or hbr_j > len(matrix[0]) - 1:
                    #    atl_mat[i][j] = True
                    elif hbr_i >= 0 and hbr_j >= 0 and matrix[hbr_i][hbr_j] <= matrix[loc_i][loc_j]:
                        if (hbr_i, hbr_j) in hs and atl_mat[hbr_i][hbr_j] == True:
                            hs.add((i, j))
                            atl_mat[i][j] = True                            
                            return True
                        elif (hbr_i, hbr_j) not in hs:
                            q.append((hbr_i, hbr_j))
        hs.add((i, j))
        atl_mat[i][j] = False
        return False        
        