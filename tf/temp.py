from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
flags.DEFINE_integer("task_index", 0,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_string("job_name", "worker", "job name worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")

FLAGS = flags.FLAGS

replicas_to_aggregate = 4
learning_rate = 0.01

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
    ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "node-0.balarjun-assgn2.uwmadison744-f18-pg0.wisc.cloudlab.us:4027"
    ],
    "worker" : [
        "node-0.balarjun-assgn2.uwmadison744-f18-pg0.wisc.cloudlab.us:4029",
        "node-1.balarjun-assgn2.uwmadison744-f18-pg0.wisc.cloudlab.us:4027",
        "node-2.balarjun-assgn2.uwmadison744-f18-pg0.wisc.cloudlab.us:4027",
        "node-3.balarjun-assgn2.uwmadison744-f18-pg0.wisc.cloudlab.us:4027"
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

def main(unused_argv):
    cluster = clusterSpec[FLAGS.deploy_mode]
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    batch_size = 100
    train_steps = 20*int(mnist.train.num_examples/batch_size) # 20 training epochs   

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)


    # TODO: Make this generic to get from cluster spec. Get the number of workers.
    num_workers = 4


    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()

    is_chief = (FLAGS.task_index == 0)
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    with tf.device(
        tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_device="/job:ps/cpu:0",
        cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

        # Set model weights - Initialize all weights to 0.
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        # Prediction Function
        prediction  = tf.nn.softmax(tf.matmul(x, W) + b)

        # Loss Function
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Gradient Descent
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=replicas_to_aggregate,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas")

        train_step = opt.minimize(loss, global_step=global_step)


        local_init_op = opt.local_step_init_op
        if is_chief:
            local_init_op = opt.chief_init_op

        ready_for_local_init_op = opt.ready_for_local_init_op

        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = opt.get_chief_queue_runner()
        sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            local_init_op=local_init_op,
            ready_for_local_init_op=ready_for_local_init_op,
            recovery_wait_secs=1,
            global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                FLAGS.task_index)

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        if is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        local_step = 0
        while True:
            # Training feed
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            train_feed = {x: batch_xs, y: batch_ys}

            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print("%f: Worker %d: training step %d done (global step: %d)" %
                (now, FLAGS.task_index, local_step, step))

            if step >= train_steps:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # Test model
        with tf.Session(server.target) as s:
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}, session=s))

if __name__ == "__main__":
  tf.app.run()
