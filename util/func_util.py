
def mfilter(pred,arr):
    result = []
    for e in arr:
        if(pred(e)):
            result.append(e)
    return result
def pipe(arg, *funcs):
    for f in funcs:
        arg = f(arg)
    return arg
def iter(fun,arr):
    for k in arr:
        fun(k)

def iteri(fun,arr):
    for i in range(len(arr)):
        fun(i,arr[i])

def init(fun,cnt):
    list = [] 
    for i in range(cnt):
        list.append(fun())
    return list

def funs_map(funs):
    r = []
    for fun in funs:
        r.append(fun())
    return r


def init(fun,cnt):
    mlist = [] 
    for i in range(cnt):
        mlist.append(fun())
    return mlist

def initi(fun,cnt):
    list = [] 
    for i in range(cnt):
        list.append(fun(i))
    return list

def initi_tuple(fun,cnt):
    a_list = [] 
    b_list = []
    for i in range(cnt):
        a,b = fun(i)
        a_list.append(a)
        b_list.append(b)
    return a_list,b_list

def initi_3tuple(fun,cnt):
    a_list = [] 
    b_list = []
    c_list = []
    for i in range(cnt):
        a,b = fun(i)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
    return a_list,b_list,c_list

def mapi_3tuple(fun,arr):
    a_list = [] 
    b_list = []
    c_list = []
    for i in range(len(arr)):
        a,b,c = fun(i,arr[i])
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
    return a_list,b_list,c_list

def map2i_3tuple(fun,arr1,arr2):
    a_list = [] 
    b_list = []
    c_list = []
    for i in range(len(arr1)):
        a,b,c = fun(i,arr1[i],arr2[i])
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
    return a_list,b_list,c_list

def mapi(fun,arr):
    a_list = [] 
    for i in range(len(arr)):
        a = fun(i,arr[i])
        a_list.append(a)
    return a_list



def mmap(fun,arr):
    return list(map(fun,arr))