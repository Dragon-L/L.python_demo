import queue
from multiprocessing.managers import BaseManager


def main():
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    BaseManager.register('get_task_queue', callable=lambda: task_queue)
    BaseManager.register('get_result_queue', callable=lambda: result_queue)

    manager = BaseManager(address=('', 5000), authkey=b'test_key')

    manager.start()

    print('try to get task queue')
    task = manager.get_task_queue()
    for i in range(10):
        print('Put taks %d' %i)
        task.put(i)

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