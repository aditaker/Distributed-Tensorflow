import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
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

if FLAGS.deploy_mode == 'cluster':
	num_workers = 2
elif FLAGS.deploy_mode == 'cluster2':
	num_workers = 3
else :
	num_workers = 1

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    #put your code here
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	is_chief = (FLAGS.task_index == 0)
	with tf.device(tf.train.replica_device_setter(ps_tasks=1, worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=clusterinfo)):
		#print('Training Data shape :' ,mnist.train.images.shape)
		#Creating the Graph
		#Declaring Input and Output
		x = tf.placeholder(tf.float32, shape=[None, 784])
		y = tf.placeholder(tf.float32, shape=[None, 10])
		#Declaring weights and biases
		w = tf.get_variable("weights", [784,10], initializer=tf.random_normal_initializer())
		b = tf.get_variable("biases", [10], initializer=tf.random_normal_initializer())

		global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
		#Defining loss function	
		prediction  = tf.nn.softmax(tf.matmul(x, w) + b)
		loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1)) 
		
		#Defining SGD optimizer
		opt = tf.train.GradientDescentOptimizer(0.01)
		optimizer = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate = num_workers, total_num_replicas = num_workers)
		optimizer_f = optimizer.minimize(loss,global_step=global_step)
		
		#Defining function to compute accuracy
		predictions_check = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy_f = tf.reduce_mean(tf.cast(predictions_check, tf.float32))
		#init = tf.global_variables_initializer()

	batch_size = 100
	n_batches = int(len(mnist.train.labels)/batch_size)
	n_epochs = 10

	local_init_op = optimizer.local_step_init_op	
	if is_chief:
		local_init_op = optimizer.chief_init_op

	ready_for_local_init_op = optimizer.ready_for_local_init_op

	chief_queue_runner = optimizer.get_chief_queue_runner()
	sync_init_op = optimizer.get_init_tokens_op()

	init = tf.global_variables_initializer()
	sv = tf.train.Supervisor(
	    is_chief=is_chief,
	    init_op=init,
	    local_init_op=local_init_op,
	    ready_for_local_init_op=ready_for_local_init_op,
	    recovery_wait_secs=1,
	    global_step=global_step)
	
	config = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=False)

	if is_chief:
		print("Worker %d: Initializing session..." % FLAGS.task_index)
	else:
		print("Worker %d: Waiting for session to be initialized..." %FLAGS.task_index)

	sess = sv.prepare_or_wait_for_session(server.target, config=config)

	print("Worker %d: Session initialization complete." % FLAGS.task_index)

	if is_chief:
	# Chief worker will start the chief queue runner and call the init op.
		sess.run(sync_init_op)
		sv.start_queue_runners(sess, [chief_queue_runner])

	time_begin = time.time()
	print('Start time of Training: ', time_begin)
	avg_cost = 0
	#for epoch in range(n_epochs):
		#t = time.time()
		#print('Start time of Epoch: ',epoch+1,' :',  t)
	step = 0
	while step < n_epochs*n_batches:
		batch_xs, batch_ys = mnist.train.next_batch(batch_size = batch_size)
		l,step, _ = sess.run([loss, global_step, optimizer_f], feed_dict={x: batch_xs, y: batch_ys})
		#print(sess.run(accuracy_f, feed_dict={x: mnist.test.images, y: mnist.test.labels}), step)	
		
		#t = time.time()
		#print('End time of Epoch: ',epoch +1,' :',  t)
		#print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
	time_end = time.time()
	print('End time of training: ', time_end)
	print('Total time taken: ' , time_end-time_begin)
	#print sess.run(accuracy_f, feed_dict={x: mnist.test.images, y: mnist.test.labels})	
	with sess.as_default():
		print("Device:", '%04d' % FLAGS.task_index, "Accuracy:", accuracy_f.eval({x: mnist.test.images, y: mnist.test.labels}))
	#tf.summary.FileWriter('./tensorBoard', sess.graph)	
