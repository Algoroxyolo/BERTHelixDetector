'''
Code Challenge (15 points): Implement SmallParsimony to solve the Small Parsimony Problem.

Input: An integer n followed by an adjacency list for a rooted binary tree with n leaves labeled by DNA strings.
Output: The minimum parsimony score of this tree, followed by the adjacency list of a tree corresponding to labeling internal nodes by DNA strings in order to minimize the parsimony score of the tree.  You may break ties however you like.
Note: Remember to run SmallParsimony on each individual index of the strings at the leaves of the tree.
Sample Input:

4
4->CAAATCCC
4->ATTGCGAC
5->CTGCGCTG
5->ATGGACGA
6->4
6->5
Sample Output:

16
ATTGCGAC->ATAGCCAC:2
ATAGACAA->ATAGCCAC:2
ATAGACAA->ATGGACTA:2
ATGGACGA->ATGGACTA:1
CTGCGCTG->ATGGACTA:4
ATGGACTA->CTGCGCTG:4
ATGGACTA->ATGGACGA:1
ATGGACTA->ATAGACAA:2
ATAGCCAC->CAAATCCC:5
ATAGCCAC->ATTGCGAC:2
ATAGCCAC->ATAGACAA:2
CAAATCCC->ATAGCCAC:5
'''

def small_parsimony(input_string):
    class Node:
        def __init__(self, label=None):
            self.label = label
            self.children = []
            self.s = [{}, {}, {}, {}]  # A, C, G, T
            self.ancestor = None

    def parse_input(input_string):
        lines = input_string.strip().split('\n')
        n = int(lines[0])
        edges = lines[1:]
        nodes = {}
        for edge in edges:
            parent, child = edge.split('->')
            if child.isdigit():
                if child not in nodes:
                    nodes[child] = Node(child)
                if parent not in nodes:
                    nodes[parent] = Node(parent)
                nodes[parent].children.append(nodes[child])
            else:
                if parent not in nodes:
                    nodes[parent] = Node(parent)
                nodes[parent].children.append(Node(child))
        return n, nodes

    def score(node, k, nucleotide):
        if k in node.s[nucleotide]:
            return node.s[nucleotide][k]
        if not node.children:
            return 0 if node.label[k] == nucleotide else float('inf')
        min_score = float('inf')
        for i, nucleotide_child in enumerate("ACGT"):
            temp_score = sum(score(child, k, nucleotide_child) + (nucleotide != nucleotide_child) for child in node.children)
            if temp_score < min_score:
                min_score = temp_score
                node.ancestor = nucleotide_child
        node.s[nucleotide][k] = min_score
        return min_score

    def small_parsimony_score(n, nodes):
        root = next(iter(nodes.values()))
        total_score = 0
        for k in range(n):
            scores = [score(root, k, nucleotide) for nucleotide in "ACGT"]
            total_score += min(scores)
        return total_score

    n, nodes = parse_input(input_string)
    return small_parsimony_score(n, nodes)

# Example usage
input_string = """
4
4->CAAATCCC
4->ATTGCGAC
5->CTGCGCTG
5->ATGGACGA
6->4
6->5
"""
print(small_parsimony(input_string))
