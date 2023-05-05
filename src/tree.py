class TreeNodeeeeeee(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

#生成先序序列的负标签树
def buildtree(tree,output_lang):

    head = TreeNodeeeeeee(tree[0])
    ans = head
    stack = []
    stack.append(head)
    for i in range(1, len(tree)):
        node = TreeNodeeeeeee(tree[i])
        if (tree[i] == output_lang['/'] or tree[i] ==output_lang['-'] or tree[i] == output_lang['+'] or tree[i] == output_lang['*'] or tree[i] == output_lang['^']):
            if (head.left == None):
                stack.append(node)
                head.left = node
                head = node
            else:
                stack.pop()
                stack.append(node)
                head.right = node
                head = node
        else:
            if (head.left == None):
                head.left = node
            else:
                if (len(stack) != 0):
                    head = stack[len(stack) - 1]
                head.right = node
                stack.pop()
                if (len(stack) != 0):
                    head = stack[len(stack) - 1]
    res1 = list()
    stack1 = []
    node1 = ans
    while stack1 or node1:
        while node1:
            res1.append(node1.val)
            stack1.append(node1)
            node1 = node1.left
        node1 = stack1.pop()
        node1 = node1.right
    res = list()
    stack = []
    node = ans
    while stack or node:
        while node:
            res.append(node.val)
            stack.append(node)
            node = node.left
        node = stack.pop()
        node = node.right
    res = list()
    res.append(head.val)
    stack = []
    node = ans
    i = 0
    while stack or node:
        while node:
            if (node.left != None):
                if (node.val == output_lang['+'] or node.val == output_lang['*']):
                    res.append(node.left.val)
                else:
                    res.append(node.right.val)
            stack.append(node)
            node = node.left
        node = stack.pop()
        if (node != None and node.left != None):
            if (node.val == output_lang['+'] or node.val == output_lang['*']):
                res.append(node.right.val)
            else:
                res.append(node.left.val)
        node = node.right
    return res