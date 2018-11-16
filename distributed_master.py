import queue
from multiprocessing.managers import BaseManager
from collections import deque


def main():
    task_queue = deque(list(range(10)))
    result_queue = queue.Queue()

    manager = BaseManager(address=('', 5000), authkey=b'test_key')

    BaseManager.register('get_task_queue', callable=lambda: task_queue)
    BaseManager.register('get_result_queue', callable=lambda: result_queue)

    manager.start()
    print('try to get result queue')
    result = manager.get_result_queue()
    print('Try to get results')
    for i in range(10):
        r = result.get(timeout = 100)
        print('Result: %s' %r)

    manager.shutdown()
    print('manager exit')


if __name__ == '__main__':
    main()