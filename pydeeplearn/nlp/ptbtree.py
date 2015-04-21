# PTB reader (Penn Treebank Project)
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

class PTBTree:
    def __init__(self, value, label, left, right, size):
        self._value = value
        self._label = label
        self._left = left
        self._right = right
        self._size = size
    
    def _shifted_str(self, level=0):
        s = ('%02d: ' % level) + '_' * (level * 4)
        if self._left is not None: # and self._right is not None
            s += ' : %s\n' % (self._label)
            s += self._left._shifted_str(level + 1)
            s += self._right._shifted_str(level + 1)
        else:
            s += '%s : %s\n' % (self._value, self._label)
        return s
    
    def __str__(self):
        return self._shifted_str(0)
    
    def get_words(self, l = []):
        if self._left is not None: # and self._right is not None
            self._left.get_words(l)
            self._right.get_words(l)
        else:
            l.append(self._value)
        return l
    
    @property
    def left(self):
        return self._left
    
    @property
    def right(self):
        return self._right
    
    @property
    def value(self):
        return self._value
    
    @property
    def label(self):
        return self._label
    
    @property
    def size(self):
        return self._size
    
    @staticmethod
    def parse(s):
        if len(s) == 0:
            return None
        return PTBTree._parse(s)[1]
    
    @staticmethod
    def _parse(s, i=0):
        j = s.find(' ', i)
        label = s[i + 1:j] # skip '('
        
        j += 1
        if s[j] == '(':
            k, left = PTBTree._parse(s, j)
            k, right = PTBTree._parse(s, k + 1) # skip ' '
            value = left._value + ' ' + right._value
            size = left._size + right._size + 1
        else:
            k = s.find(')', j)
            value = s[j:k]
            left = right = None
            size = 1
        
        k += 1 # skip ')'
        return k, PTBTree(value, label, left, right, size)
