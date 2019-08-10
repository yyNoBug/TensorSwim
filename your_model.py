from executor import *
from functional import *


class Session(object):
    def __call__(self, name="Session"):
        newSession = Session()
        newSession.name = name
        newSession.ex = None   # I don't know what it means
        return newSession

    def run(self, eval_node_list, feed_dict={}):
        if isinstance(eval_node_list, list):
            executor = Executor(eval_node_list)
            return executor.run(feed_dict=feed_dict)
        else:
            executor = Executor([eval_node_list])
            return executor.run(feed_dict=feed_dict)[0]

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        return
