import multiprocessing
import time
import redis
import sys
import numpy as np


def pub(myredis):
    for n in range(3):
        myredis.publish('channel', 'blah %d' % n)
        time.sleep(5)
    temp_np = np.random.randn(10000).reshape(1000, 10)

    array_dtype = str(temp_np.dtype)
    l, w = temp_np.shape
    temp_np = temp_np.ravel().tostring()

    key = '{0}|{1}#{2}#{3}'.format(0, array_dtype, l, w)

    myredis.set('key', key)
    myredis.set('array', temp_np)

    #myredis.publish('key', temp_np)
    for n in range(3,6):
        myredis.publish('channel', 'blah %d' % n)
        time.sleep(5)



def sub(myredis, name):
    pubsub = myredis.pubsub()
    pubsub.subscribe(['channel'])
    for item in pubsub.listen():
        print('%s : %s' % (name, item['data']))

def subnp(myredis, name):

    print('here we should retrieve the array')

def main(argv):
    myredis = redis.Redis(db=0)
    myredis2 = redis.Redis(db=1)

    temp_np = np.random.randn(10000).reshape(1000, 10)

    array_dtype = str(temp_np.dtype)
    l, w = temp_np.shape
    temp_np = temp_np.ravel().tostring()

    key = '{0}|{1}#{2}#{3}'.format(0, array_dtype, l, w)

    myredis.set('key', key)
    myredis.set('array', temp_np)

    print('check here')

    """multiprocessing.Process(target=pub, args=(myredis,)).start()
    multiprocessing.Process(target=pub, args=(myredis2,)).start()
    multiprocessing.Process(target=sub, args=(myredis, 'reader 1')).start()
    multiprocessing.Process(target=sub, args=(myredis2, 'reader 2')).start()
    multiprocessing.Process(target=subnp, args=(myredis, 'reader 3')).start()
    multiprocessing.Process(target=subnp, args=(myredis, 'reader 4')).start()"""

    print('stop here')


main(sys.argv[1:])
