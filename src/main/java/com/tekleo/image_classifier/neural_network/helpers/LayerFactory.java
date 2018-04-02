package com.tekleo.image_classifier.neural_network.helpers;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Simple factory methods to create layers for CNN with more ease
 *
 * @author Leo Ertuna
 * @since 02.04.2018 12:49
 */
public class LayerFactory {
    public static ConvolutionLayer convolutionLayer(int kernelSize, int stride, int nIn, int nOut, Activation activation) {
        return new ConvolutionLayer.Builder()
                .kernelSize(kernelSize, kernelSize)
                .stride(stride, stride)
                .nIn(nIn)
                .nOut(nOut)
                .activation(activation)
                .build();
    }

    public static ConvolutionLayer convolutionLayer(int kernelSize, int stride, int nOut, Activation activation) {
        return new ConvolutionLayer.Builder()
                .kernelSize(kernelSize, kernelSize)
                .stride(stride, stride)
                .nOut(nOut)
                .activation(activation)
                .build();
    }

    public static SubsamplingLayer subsamplingLayer(int kernelSize, int stride, SubsamplingLayer.PoolingType poolingType) {
        return new SubsamplingLayer.Builder()
                .poolingType(poolingType)
                .kernelSize(kernelSize, kernelSize)
                .stride(stride, stride)
                .build();
    }

    public static DenseLayer denseLayer(int nOut, Activation activation) {
        return new DenseLayer.Builder()
                .activation(activation)
                .nOut(nOut)
                .build();
    }

    public static OutputLayer outputLayer(int nOut, Activation activation, LossFunctions.LossFunction lossFunction) {
        return new OutputLayer.Builder()
                .lossFunction(lossFunction)
                .nOut(nOut)
                .activation(activation)
                .build();
    }
}
