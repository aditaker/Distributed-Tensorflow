import tensorflow as tf
import time
# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

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
        "10.10.1.1:2222"
    ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222",
        "10.10.1.3:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    #put your code here

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=clusterinfo)):
		print('Training Data shape :' ,mnist.train.images.shape)
		x = tf.placeholder(tf.float32, shape=[None, 784])
		y = tf.placeholder(tf.float32, shape=[None, 10])
		w = tf.get_variable("weights", [784,10], initializer=tf.random_normal_initializer())
		b = tf.get_variable("biases", [10], initializer=tf.random_normal_initializer())
		prediction  = tf.nn.softmax(tf.matmul(x, w) + b)
		loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1)) 
		optimizer = tf.train.GradientDescentOptimizer
		optimizer_f = optimizer(learning_rate=0.01).minimize(loss)
		predictions_check = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy_f = tf.reduce_mean(tf.cast(predictions_check, tf.float32))
		init = tf.global_variables_initializer()
	batch_size = 100
	n_batches = int(len(mnist.train.labels)/batch_size)
	n_epochs = 10
	sess = tf.Session(target=server.target)
	sess.run(init)
	time_begin = time.time()
	print('Start time of Training: ', time_begin)
	for epoch in range(n_epochs):
		avg_cost = 0
		for batch in range(n_batches):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size = batch_size)
			l, _ = sess.run([loss, optimizer_f], feed_dict={x: batch_xs, y: batch_ys})
			avg_cost += l/n_batches
		print("Device:", '%04d' % FLAGS.task_index,"Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
	time_end = time.time()
	print('End time of Training: ',  time_end)
	print('Total time taken: ', time_end-time_begin)
	with sess.as_default():
		print("Device:", '%04d' % FLAGS.task_index, "Accuracy:", accuracy_f.eval({x: mnist.test.images, y: mnist.test.labels}))
	tf.summary.FileWriter('./tensorBoard/example', sess.graph)
