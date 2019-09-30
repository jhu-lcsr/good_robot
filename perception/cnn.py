"""
Wrapper for tensorflow Convolutional Neural Network classes
Author: Jeff Mahler
"""
import numpy as np
import os
import tensorflow as tf

class AlexNetWeights(object):
    """ Struct helper for storing weights """
    def __init__(self):
        pass

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    """
    Convolution layer helper function
    From https://github.com/ethereon/caffe-tensorflow
    """
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, axis=3)
        kernel_groups = tf.split(kernel, group, axis=3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, axis=3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

class AlexNet(object):
    """ Wrapper for tensorflow AlexNet. Note: training not yet supported.

    Parameters
    ----------
    config : :obj:`autolab_core.YamlConfig`
        specifies the parameters of the network

    Notes
    -----
    Required configuration paramters are specified in Other Parameters

    Other Parameters
    ----------------
    batch_size : int
        size of batches, less than largest possible prediction to save memory
    im_height : int
        height of input images
    im_width : int
        width of input images
    channels : int
        number of channels of input image (should be 3)
    output_layer : :obj:`str`
        name of output layer for classification
    feature_layer : :obj`str`
        name of layer to use for feature extraction (e.g. conv5)
    """
    def __init__(self, config, model_dir=None, use_default_weights=False,
                 dynamic_load=True):
        self._model_dir = model_dir
        self._sess = None
        self._initialized = False
        self._dynamic_load = dynamic_load
        self._parse_config(config)
        if use_default_weights:
            self._initialize()
        elif not self._dynamic_load:
            self._load()

    def _parse_config(self, config):
        """ Parses a tensorflow configuration """
        self._batch_size = config['batch_size']
        self._im_height = config['im_height']
        self._im_width = config['im_width']
        self._num_channels = config['channels']
        self._output_layer = config['out_layer']
        self._feature_layer = config['feature_layer']
        self._out_size = None
        if 'out_size' in config.keys():
            self._out_size = config['out_size']
        self._input_arr = np.zeros([self._batch_size, self._im_height,
                                    self._im_width, self._num_channels])

        if self._model_dir is None:
            self._net_data = np.load(config['caffe_weights']).item()
            self._mean = np.load(config['mean_file'])
            self._model_filename = None
        else:
            self._net_data = None
            self._mean = np.load(os.path.join(self._model_dir, 'mean.npy'))
            self._model_filename = os.path.join(self._model_dir, 'model.ckpt')

    def _load(self):
        """ Loads a model into weights """
        if self._model_filename is None:
            raise ValueError('Model filename not specified')

        # read the input image
        self._graph = tf.Graph()
        with self._graph.as_default():
            # read in filenames
            reader = tf.train.NewCheckpointReader(self._model_filename)

            # load AlexNet weights
            weights = AlexNetWeights()
            weights.conv1W = tf.Variable(reader.get_tensor("Variable")) 
            weights.conv1b = tf.Variable(reader.get_tensor("Variable_1"))
            weights.conv2W = tf.Variable(reader.get_tensor("Variable_2"))
            weights.conv2b = tf.Variable(reader.get_tensor("Variable_3"))
            weights.conv3W = tf.Variable(reader.get_tensor("Variable_4"))
            weights.conv3b = tf.Variable(reader.get_tensor("Variable_5"))
            weights.conv4W = tf.Variable(reader.get_tensor("Variable_6"))
            weights.conv4b = tf.Variable(reader.get_tensor("Variable_7"))
            weights.conv5W = tf.Variable(reader.get_tensor("Variable_8"))
            weights.conv5b = tf.Variable(reader.get_tensor("Variable_9"))
            weights.fc6W = tf.Variable(reader.get_tensor("Variable_10"))
            weights.fc6b = tf.Variable(reader.get_tensor("Variable_11"))
            weights.fc7W = tf.Variable(reader.get_tensor("Variable_12"))
            weights.fc7b = tf.Variable(reader.get_tensor("Variable_13"))
            weights.fc8W = tf.Variable(reader.get_tensor("Variable_14"))
            weights.fc8b = tf.Variable(reader.get_tensor("Variable_15"))

            # form network
            self._input_node = tf.placeholder(tf.float32, (self._batch_size, self._im_height, self._im_width, self._num_channels))
            self._output_tensor = self.build_alexnet(weights)
            self._feature_tensor = self.build_alexnet(weights, output_layer=self._feature_layer)
            self._initialized = True

    def _initialize(self):
        """ Open from caffe weights """
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._input_node = tf.placeholder(tf.float32, (self._batch_size, self._im_height, self._im_width, self._num_channels))
            weights = self.build_alexnet_weights()
            self._output_tensor = self.build_alexnet(weights)
            self._feature_tensor = self.build_alexnet(weights, output_layer=self._feature_layer)
            self._initialized = True

    def open_session(self):
        """ Open tensorflow session. Exposed for memory management. """
        with self._graph.as_default():
            init = tf.initialize_all_variables()
            self._sess = tf.Session()
            self._sess.run(init)

    def close_session(self):
        """ Close tensorflow session. Exposes for memory management. """
        with self._graph.as_default():
            self._sess.close()
            self._sess = None

    def predict(self, image_arr, featurize=False):
        """ Predict a set of images in batches.

        Parameters
        ----------
        image_arr : NxHxWxC :obj:`numpy.ndarray`
            input set of images in a num_images x image height x image width x image channels array (must match parameters of network)
        featurize : bool
            whether or not to use the featurization layer or classification output layer

        Returns
        -------
        :obj:`numpy.ndarray`
            num_images x feature_dim containing the output values for each input image
        """
        # setup prediction
        num_images = image_arr.shape[0]
        output_arr = None

        # predict by filling in image array in batches
        close_sess = False
        if not self._initialized and self._dynamic_load:
            self._load()
        with self._graph.as_default():
            if self._sess is None: 
                close_sess = True               
                self.open_session()

            i = 0
            while i < num_images:
                dim = min(self._batch_size, num_images-i)
                cur_ind = i
                end_ind = cur_ind + dim
                self._input_arr[:dim,:,:,:] = image_arr[cur_ind:end_ind,:,:,:] - self._mean
                if featurize:
                    output = self._sess.run(self._feature_tensor,
                                            feed_dict={self._input_node: self._input_arr})
                else:
                    output = self._sess.run(self._output_tensor,
                                            feed_dict={self._input_node: self._input_arr})
                if output_arr is None:
                    output_arr = output
                else:
                    output_arr = np.r_[output_arr, output]

                i = end_ind
            if close_sess:
                self.close_session()
        return output_arr[:num_images,...]

    def featurize(self, image_arr):
        """ Featurize a set of images in batches.

        Parameters
        ----------
        image_arr : NxHxWxC :obj:`numpy.ndarray`
            input set of images in a num_images x image height x image width x image channels array (must match parameters of network)

        Returns
        -------
        :obj:`numpy.ndarray`
            num_images x feature_dim containing the output values for each input image
        """
        return self.predict(image_arr, featurize=True)

    def build_alexnet_weights(self):
        """ Build a set of convnet weights for AlexNet """
        net_data = self._net_data
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
  
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
    
        #fc6
        #fc(4096, name='fc6')
        fc6_in_size = net_data["fc6"][0].shape[0]
        fc6_out_size = net_data["fc6"][0].shape[1]
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])

        #fc7
        #fc(4096, name='fc7') 
        fc7_in_size = fc6_out_size
        fc7_out_size = net_data["fc7"][0].shape[1]
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
    
        #fc8
        #fc(num_cats, relu=False, name='fc8')                       
        fc8_in_size = fc7_out_size
        fc8_out_size = self._out_size
        fc8W = tf.Variable(tf.truncated_normal([fc8_in_size, fc8_out_size],
                                               stddev=0.01,
                                               seed=None))
        fc8b = tf.Variable(tf.constant(0.0, shape=[fc8_out_size]))
  
        # make return object
        weights = AlexNetWeights()
        weights.conv1W = conv1W
        weights.conv1b = conv1b
        weights.conv2W = conv2W
        weights.conv2b = conv2b
        weights.conv3W = conv3W
        weights.conv3b = conv3b 
        weights.conv4W = conv4W
        weights.conv4b = conv4b 
        weights.conv5W = conv5W
        weights.conv5b = conv5b 
        weights.fc6W = fc6W
        weights.fc6b = fc6b
        weights.fc7W = fc7W
        weights.fc7b = fc7b
        weights.fc8W = fc8W
        weights.fc8b = fc8b
        return weights

    def build_alexnet(self, weights, output_layer=None):
        """ Connects graph of alexnet from weights """
        if output_layer is None:
            output_layer = self._output_layer

        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1_in = conv(self._input_node, weights.conv1W, weights.conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
        if output_layer == 'conv1':
            return tf.contrib.layers.flatten(conv1)
        
        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        
        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2_in = conv(maxpool1, weights.conv2W, weights.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        if output_layer == 'conv2':
            return tf.contrib.layers.flatten(conv2)
                
        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        
        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3_in = conv(maxpool2, weights.conv3W, weights.conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        if output_layer == 'conv3':
            return tf.contrib.layers.flatten(conv3)
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4_in = conv(conv3, weights.conv4W, weights.conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        if output_layer == 'conv4':
            return tf.contrib.layers.flatten(conv4)
        
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5_in = conv(conv4, weights.conv5W, weights.conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        if output_layer == 'conv5':
            return tf.contrib.layers.flatten(conv5)
        
        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        #fc6
        #fc(4096, name='fc6')
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), weights.fc6W, weights.fc6b)
        if output_layer == 'fc6':
            return fc6
            
        #fc7
        #fc(4096, name='fc7') 
        fc7 = tf.nn.relu_layer(fc6, weights.fc7W, weights.fc7b)
        if output_layer == 'fc7':
            return fc7
    
        #fc8
        #fc(num_cats, relu=False, name='fc8')
        fc8 = tf.nn.xw_plus_b(fc7, weights.fc8W, weights.fc8b)
        if output_layer == 'fc8':
            return fc8        

        #softmax
        sm = tf.nn.softmax(fc8)
        return sm

