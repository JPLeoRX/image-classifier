package com.tekleo.image_classifier.neural_network.config;

import java.io.Serializable;
import java.util.Objects;

/**
 * Configuration of convolution layers for the network
 * This will include convolution and subsampling layers
 * The network will use the same config across all layers
 *
 * Immutable object
 *
 * @author Leo Ertuna
 * @since 01.04.2018 17:40
 */
public class ConvolutionConfig implements Serializable, Cloneable {
    private KernelConfig convolutionLayerKernelConfig;
    private KernelConfig subsamplingLayerKernelConfig;

    // Constructors
    //------------------------------------------------------------------------------------------------------------------
    /**
     * Private default constructor to prevent instantiating this object
     */
    private ConvolutionConfig() {
        throw new ConfigRuntimeException();
    }

    /**
     * Constructor from 2 given kernel configs
     * @param convolutionLayerKernelConfig convolution layer kernel config
     * @param subsamplingLayerKernelConfig subsampling layer kernel config
     */
    public ConvolutionConfig(KernelConfig convolutionLayerKernelConfig, KernelConfig subsamplingLayerKernelConfig) {
        this.convolutionLayerKernelConfig = convolutionLayerKernelConfig;
        this.subsamplingLayerKernelConfig = subsamplingLayerKernelConfig;
    }

    /**
     * For an easier usage when manually instantiating we provide direct constructor with 4 integers
     * @param convolutionLayerKernelSize convolution layer kernel size (size of the kernel matrix)
     * @param convolutionLayerKernelStride convolution layer kernel stride (shifts between each application of the kernel matrix)
     * @param subsamplingLayerKernelSize subsampling layer kernel size (size of the kernel matrix)
     * @param subsamplingLayerKernelStride subsampling layer kernel stride (shifts between each application of the kernel matrix)
     */
    public ConvolutionConfig(int convolutionLayerKernelSize, int convolutionLayerKernelStride, int subsamplingLayerKernelSize, int subsamplingLayerKernelStride) {
        this(new KernelConfig(convolutionLayerKernelSize, convolutionLayerKernelStride), new KernelConfig(subsamplingLayerKernelSize, subsamplingLayerKernelStride));
    }
    //------------------------------------------------------------------------------------------------------------------



    // Getters
    //------------------------------------------------------------------------------------------------------------------
    public int getConvolutionLayerKernelSize() {
        return convolutionLayerKernelConfig.getSize();
    }

    public int getConvolutionLayerKernelStride() {
        return convolutionLayerKernelConfig.getStride();
    }

    public int getSubsamplingLayerKernelSize() {
        return subsamplingLayerKernelConfig.getSize();
    }

    public int getSubsamplingLayerKernelStride() {
        return subsamplingLayerKernelConfig.getStride();
    }
    //------------------------------------------------------------------------------------------------------------------



    // Others
    //------------------------------------------------------------------------------------------------------------------
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ConvolutionConfig that = (ConvolutionConfig) o;
        return Objects.equals(convolutionLayerKernelConfig, that.convolutionLayerKernelConfig) && Objects.equals(subsamplingLayerKernelConfig, that.subsamplingLayerKernelConfig);
    }

    @Override
    public int hashCode() {
        return Objects.hash(convolutionLayerKernelConfig, subsamplingLayerKernelConfig);
    }

    @Override
    public String toString() {
        return "ConvolutionConfig{" + "convolutionLayerKernelConfig=" + convolutionLayerKernelConfig + ", subsamplingLayerKernelConfig=" + subsamplingLayerKernelConfig + '}';
    }

    @Override
    public ConvolutionConfig clone() {
        return new ConvolutionConfig(convolutionLayerKernelConfig.clone(), subsamplingLayerKernelConfig.clone());
    }
    //------------------------------------------------------------------------------------------------------------------
}