import time, threading


def task():
    print('Run thread %s' % threading.current_thread().name)
    start = time.time()
    time.sleep(1)
    end = time.time()
    print('Task runs %f seconds' % (end - start))
    print('Thread %s is closed' % threading.current_thread().name)


def new_thread_demo():
    print('Thread %s is running' % threading.current_thread().name)
    thread = threading.Thread(target=task, name='sub_thread')
    thread.start()
    thread.join()
    print('Thread %s is closed' % threading.current_thread().name)


new_thread_demo()