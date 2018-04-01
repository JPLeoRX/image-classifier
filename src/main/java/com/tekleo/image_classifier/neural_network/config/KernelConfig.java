package com.tekleo.image_classifier.neural_network.config;

import java.io.Serializable;
import java.util.Objects;

/**
 * Configuration of convolution or subsampling kernel. We assume that a square shape is always used.
 *
 * Immutable object
 *
 * @author Leo Ertuna
 * @since 01.04.2018 17:40
 */
public class KernelConfig implements Serializable, Cloneable {
    private int size;
    private int stride;

    // Constructors
    //------------------------------------------------------------------------------------------------------------------
    /**
     * Private default constructor to prevent instantiating this object
     */
    private KernelConfig() {
        throw new ConfigRuntimeException();
    }

    /**
     * The only public constructor, this object is immutable
     * @param size size (size of the kernel matrix)
     * @param stride stride (shifts between each application of the kernel matrix)
     */
    public KernelConfig(int size, int stride) {
        this.size = size;
        this.stride = stride;
    }
    //------------------------------------------------------------------------------------------------------------------



    // Getters
    //------------------------------------------------------------------------------------------------------------------
    public int getSize() {
        return size;
    }

    public int getStride() {
        return stride;
    }
    //------------------------------------------------------------------------------------------------------------------



    // Others
    //------------------------------------------------------------------------------------------------------------------
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KernelConfig that = (KernelConfig) o;
        return size == that.size && stride == that.stride;
    }

    @Override
    public int hashCode() {
        return Objects.hash(size, stride);
    }

    @Override
    public String toString() {
        return "KernelConfig{" + "size=" + size + ", stride=" + stride + '}';
    }

    @Override
    public KernelConfig clone() {
        return new KernelConfig(size, stride);
    }
    //------------------------------------------------------------------------------------------------------------------
}