import queue
from multiprocessing.managers import BaseManager

import time


def excute_task():
    BaseManager.register('get_task_queue')
    BaseManager.register('get_result_queue')
    manager = BaseManager(address=('127.0.0.1', 5000), authkey=b'test_key')
    manager.connect()
    task = manager.get_task_queue()
    result = manager.get_result_queue()
    for i in range(10):
        try:
            n = task.popleft()
            print('run task %d' % n)
            time.sleep(1)
            result.put(n * 2)
        except queue.Queue.Empty:
            print('task queue is empty')
    print('work exit')


if __name__ == '__main__':
    excute_task()