import os, time
from multiprocessing import Process, Pool
from concurrent.futures import ProcessPoolExecutor


def task(name):
    print('Run child process %s (%s)' % (name, os.getpid()))
    start = time.time()
    time.sleep(1)
    end = time.time()
    print('Task %s runs %f seconds' % (name, end - start))


def fork_demo():
    print('Process %s start...' %os.getpid())
    pid = os.fork()
    if pid == 0:
        print('This is child process %s, and parent process is %s' % (os.getpid(), os.getppid()))
    else:
        print('This is parent process %s' %os.getpid())


def multiprocessing_demo():
    print('Process %s start...' % os.getpid())
    process = Process(target=task, args=('child_process',))
    process.start()
    print('process is started')
    process.join()
    print('process is joined')


def pool_demo(parallel_num, total_num):
    pool = Pool(parallel_num)
    for i in range(total_num):
        pool.apply_async(task, args=('task_'+str(i),))
    # pool.map(task, ['task_1', 'task_2', 'task_3', 'task_4', 'task_6'])
    pool.close()
    pool.join()
    print('All process is closed!')


def process_executor_demo():
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(task, ['1', '2', '3', '4', '5', '6'])




# fork_demo()
# multiprocessing_demo()
# pool_demo(3, 5)
process_executor_demo()