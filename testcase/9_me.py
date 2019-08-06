import your_model as act_tf
import tensorflow as ans_tf
import numpy as np


def forward(tf):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[3, 4, 4, 2])
        w = tf.Variable(np.arange(3 * 3 * 2 * 5, dtype=np.float32).reshape(3, 3, 2, 5))

        conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

        loss = tf.reduce_sum(conv1)
        train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

        sess.run(tf.global_variables_initializer())
        sess.run(train_op, feed_dict={
            x: np.arange(3 * 4 * 4 * 2).reshape(3, 4, 4, 2),
        })

        res = w.eval()
        return res


res = forward(act_tf)
ans = forward(ans_tf)
print(res)
print(ans)
np.testing.assert_allclose(res, ans, atol=1e-2)