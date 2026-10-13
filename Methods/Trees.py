import torch
from General_methods import load_data, add_ones, tt_split, test, MSE
#TODO Tree pruning
#TODO more efficient cost function (it's convex isn't it?)

#TreeNode for building our decision tree
#Evaluation will go something like "Is data left or right, okay, if no left node return left_rule, etc"
class TreeNode():
    def __init__(self, index = None, value = None, left = None, right = None, left_rule = None, right_rule = None):
        self.index = index
        self.value = value
        self.left = left
        self.right = right
        self.left_rule = left_rule
        self.right_rule = right_rule

    def print(self):
        print(f"Index is {self.index}")
        print(f"Split value is {self.value}")
        print(f"Left rule is {self.left_rule}")
        print(f"Right rule is {self.right_rule}")

#Cost function
def cost(X_j, Y, s):
    left_mask = X_j <= s
    Y_left = Y[left_mask]
    Y_right = Y[~left_mask]

    c_left = Y_left.mean()
    c_right = Y_right.mean()

    cost_left = torch.sum((Y_left - c_left)**2)
    cost_right = torch.sum((Y_right - c_right)**2)

    return (cost_left + cost_right, c_left, c_right)

synthetic_data = torch.tensor([[0.1, 0.9, 5.1], [0.2, 0.8, 5.3], [0.1, 0.8, 5.6], [0.2, 0.9, 4.9],
                  [0.1, 0.1, 1], [0.1, 0.2, 1.1],
                  [0.9, 0.1, 3.1], [0.8, 0.2, 3.3], [0.9, 0.2, 3.6], [0.8, 0.1, 2.7],
                  [0.9, 0.9, 2], [0.8, 0.8, 2.2]])

#We now implement the greedy algorithm described on page 307 of ESL
#min_j,s[min_c1 sum (y_i-c_1)^2 + min_c2 sum (y_i-c_2)^2]
#summing over x_i in R_1 and x_i in R_2 respectively.

#I guess for practical implementation it will be best to search each dimension sequentially
#A gpu could do them all concurrently which is cool

#Split to X and Y values
X = synthetic_data[:,:2]
Y = synthetic_data[:, -1]

#Recursive function to be called
def make_tree(X, Y, max_depth = 3):
    if max_depth == 0:
        return None
    #Get shape of X for loops
    n, m = X.shape
    #Loop to choose split point (to be functionalized)
    #Probably bad to use discrete ranges for the data splits, but its much more computationally feasible
    #than, say, finding the midpoints of all the data to use. 
    min_cost = torch.inf
    best_split = (0,0)
    c_left = 0
    c_right = 0
    for j in range(m):
        #number of regions to check
        #TODO dynamic values to be check
        #TODO control for number of slices checked
        for i in range(10):
            val = i/10 + 0.05
            curr = cost(X[:, j], Y, val)
            if curr[0] < min_cost:
                min_cost = curr[0]
                best_split = (j, val)
                c_left = curr[1]
                c_right = curr[2]



    #split data and recur
    left_mask = X[:, best_split[0]] < best_split[1]
    X_left = X[left_mask]
    Y_left = Y[left_mask]
    X_right = X[~left_mask]
    Y_right = Y[~left_mask]

    #At this point call the recursive function on (X_left, Y_left) and (X_right, Y_right)
    #But we need logic for putting the decisions into the tree we made above
    new_node = TreeNode(index = best_split[0], 
                        value = best_split[1],
                        left = None,
                        right = None,
                        left_rule=c_left,
                        right_rule=c_right)
    
    new_node.left = make_tree(X_left, Y_left, max_depth=max_depth-1)
    new_node.right = make_tree(X_right, Y_right, max_depth=max_depth-1)

    return new_node

tree = make_tree(X, Y, 2)
tree.right.print()


