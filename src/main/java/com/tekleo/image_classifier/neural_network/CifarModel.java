package com.tekleo.image_classifier.neural_network;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

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
 *  Accuracy:        0.7005
 *  Precision:       0.7076
 *  Recall:          0.7005
 *  F1 Score:        0.7021
 *
 * Due to CUDA support in DL4J we can utilize GPU training:
 * GPU load in training varies from 65% to 95%
 * While CPU load alternates between 30% and 50%
 * Current model consumes about 1.4 GB of GPU memory
 *
 * This is achieved on i7, gtx1050 (4 GB) and 16 GB ram
 *
 * 99.48/69.15
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
    private static final String NETWORK_FILEPATH = "CIFAR-10 Network.zip";

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

    private static Layer convLayer(int nIn, int nOut) {
        return new ConvolutionLayer.Builder()
                .kernelSize(CONVOLUTION_FILTER_SIZE, CONVOLUTION_FILTER_SIZE)
                .stride(CONVOLUTION_FILTER_SHIFT, CONVOLUTION_FILTER_SHIFT)
                .nIn(nIn)
                .nOut(nOut)
                .activation(Activation.IDENTITY)
                .build();
    }

    private static Layer convLayer(int nOut) {
        return new ConvolutionLayer.Builder()
                .kernelSize(CONVOLUTION_FILTER_SIZE, CONVOLUTION_FILTER_SIZE)
                .stride(CONVOLUTION_FILTER_SHIFT, CONVOLUTION_FILTER_SHIFT)
                .nOut(nOut)
                .activation(Activation.IDENTITY)
                .build();
    }

    private static Layer downsamplingLayer() {
        return new SubsamplingLayer.Builder()
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(CONVOLUTION_POOL_SIZE, CONVOLUTION_POOL_SIZE)
                .stride(CONVOLUTION_POOL_SHIFT, CONVOLUTION_POOL_SHIFT)
                .build();
    }

    private void initConfig() {
        configuration = new NeuralNetConfiguration.Builder()
                .seed(NETWORK_SEED)
                .iterations(LEARNING_NUMBER_OF_ITERATIONS)

                .regularization(true).l2(1e-4)

                .learningRate(1e-2).learningRateDecayPolicy(LearningRatePolicy.Score).lrPolicyDecayRate(1e-3)

                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))

                .list()


                // After layer 0 we have: 32 x 32 x 10
                .layer(0, convLayer(IMAGE_CHANNELS, 64))

                // After layer 1 we have: 16 x 16 x 10
                .layer(1, downsamplingLayer())

                // After layer 2 we have: 16 x 16 x 20
                .layer(2, convLayer(128))

                // After layer 3 we have 8 x 8 x 20
                .layer(3, downsamplingLayer())

                // MLP
                .layer(4, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(4800)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(1200)
                        .build())
                .layer(6, new OutputLayer.Builder()
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
        Evaluation evaluationOnTrain = network.evaluate(trainSet);
        System.out.println(evaluationOnTrain.stats());

        Evaluation evaluationOnTest = network.evaluate(testSet);
        System.out.println(evaluationOnTest.stats());
    }

    public void save() throws IOException {
        File locationToSave = new File(NETWORK_FILEPATH);
        ModelSerializer.writeModel(network, locationToSave, true);
    }

    public void load() throws IOException {
        File locationToSave = new File(NETWORK_FILEPATH);
        network = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
    }
}
