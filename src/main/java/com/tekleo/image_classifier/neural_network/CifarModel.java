package com.tekleo.image_classifier.neural_network;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Core CNN model
 *
 * We utilize doubled convolution-downsample architecture with regular MLP attached to the end of it
 * CONV (INPUT) - DOWNSAMPLE - CONV - DOWNSAMPLE - DENSE - DENSE (OUTPUT)
 *
 * Convolutional filter is 3 x 3 with shift (stride) of 1
 * Downsampling is a 2 x 2 kernel with shift (stride) of 2
 *
 * TODO possibly add more layers
 * TODO implement some cross-validation training
 * TODO print out some sort of training set score for each epoch
 * TODO move neural net config into a separate file
 *
 * With current setting we achieve the following scores:
 *  Accuracy:        0.6528
 *  Precision:       0.6574
 *  Recall:          0.6528
 *  F1 Score:        0.6531
 *
 * @author Leo Ertuna
 * @since 01.04.2018 02:29
 */
public class CifarModel {
    private static final int IMAGE_HEIGHT = 32;
    private static final int IMAGE_WIDTH = 32;
    private static final int IMAGE_CHANNELS = 3;
    private static final int IMAGE_CLASSES = 10;

    private static final int NETWORK_SEED = 666;

    private static final int CONVOLUTION_FILTER_SIZE = 3;
    private static final int CONVOLUTION_FILTER_SHIFT = 1;
    private static final int CONVOLUTION_POOL_SIZE = 2;
    private static final int CONVOLUTION_POOL_SHIFT = 2;

    private static final int LEARNING_NUMBER_OF_ITERATIONS = 1;
    private static final int LEARNING_NUMBER_OF_EPOCHS = 20;

    private DataSetIterator trainSet;
    private DataSetIterator testSet;
    private MultiLayerConfiguration configuration;
    private MultiLayerNetwork network;

    public CifarModel(DataSetIterator trainSet, DataSetIterator testSet) {
        this.trainSet = trainSet;
        this.testSet = testSet;
        this.initConfig();
        this.initNetwork();
    }

    private void initConfig() {
        configuration = new NeuralNetConfiguration.Builder()
                .seed(NETWORK_SEED)
                .iterations(LEARNING_NUMBER_OF_ITERATIONS)

                .regularization(true)
                .l2(1e-3)

                .learningRate(1e-2)
                .learningRateDecayPolicy(LearningRatePolicy.Score)
                .lrPolicyDecayRate(1e-3)

                .weightInit(WeightInit.XAVIER_UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))

                .list()


                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(CONVOLUTION_FILTER_SIZE, CONVOLUTION_FILTER_SIZE)
                        .stride(CONVOLUTION_FILTER_SHIFT, CONVOLUTION_FILTER_SHIFT)
                        .nIn(IMAGE_CHANNELS)
                        .nOut(12)
                        .activation(Activation.IDENTITY)
                        .build())

                /*
                 * Layer 0 is a conv layer
                 * After layer 0 we have: 32 x 32 x 12
                 */

                .layer(1, new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(CONVOLUTION_POOL_SIZE, CONVOLUTION_POOL_SIZE)
                        .stride(CONVOLUTION_POOL_SHIFT, CONVOLUTION_POOL_SHIFT)
                        .build())

                /*
                 * Layer 1 is a downsampling layer
                 * After layer 1 we have: 16 x 16 x 12
                 */

                .layer(2, new ConvolutionLayer.Builder()
                        .kernelSize(CONVOLUTION_FILTER_SIZE, CONVOLUTION_FILTER_SIZE)
                        .stride(CONVOLUTION_FILTER_SHIFT, CONVOLUTION_FILTER_SHIFT)
                        .nOut(24)
                        .activation(Activation.IDENTITY)
                        .build())

                /*
                 * Layer 2 is a conv layer
                 * After layer 2 we have: 16 x 16 x 24
                 */

                .layer(3, new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(CONVOLUTION_POOL_SIZE, CONVOLUTION_POOL_SIZE)
                        .stride(CONVOLUTION_POOL_SHIFT, CONVOLUTION_POOL_SHIFT)
                        .build())

                /*
                 * Layer 3 is a downsampling layer
                 * After layer 3 we have 8 x 8 x 24
                 */

                .layer(4, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(600)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(IMAGE_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
                .backprop(true)
                .pretrain(false)
                .build();
    }

    private void initNetwork() {
        network = new MultiLayerNetwork(configuration);
        network.init();
    }

    public void train() {
        for (int i = 0; i < LEARNING_NUMBER_OF_EPOCHS; i++) {
            network.fit(trainSet);
            System.out.println("Completed epoch " + i);
        }
    }

    public void test() {
        Evaluation evaluation = network.evaluate(testSet);
        System.out.println(evaluation.stats());
    }
}
