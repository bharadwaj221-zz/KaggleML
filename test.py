N = 200


class TrieNode:
    def __init__(self):
        self.children = [None] * N
        self.counter = 0
        self.end = False


def char_to_int(c):
    return ord(c) - ord('A')


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert_word(self, word):

        curr = self.root
        curr.counter += 1
        n = len(word)
        for i in range(n):

            c = word[i]
            k = char_to_int(c)
            # print c,k
            if not curr.children[k]:
                curr.children[k] = TrieNode()
                curr.children[k].counter = 1
            else:
                curr.children[k].counter += 1

            if i == n - 1:
                curr.children[k].end = True

            curr = curr.children[k]

    def common_prefix(self, node, n, res):
        children = node.children
        index, count = self.count_children(children, n)
        while count == 1:
            res[0] = res[0] + chr(ord('A') + index)
            children = children[index].children
            index, count = self.count_children(children, n)

    def count_children(self, children, n):
        c = 0
        index = -1
        for i in range(N):
            if children[i] and children[i].counter == n:
                # print children[i].counter
                c += 1
                index = i

        return index, c


def longestCommonPrefix(A):
        t = Trie()
        for s in A:
            t.insert_word(s)
        res = ['']
        t.common_prefix(t.root,  len(A),res)

        return res
A=[ "aaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" ]
#, "abcd", "efgh" ]
print len(A)
s=longestCommonPrefix(A)

print "---",s

