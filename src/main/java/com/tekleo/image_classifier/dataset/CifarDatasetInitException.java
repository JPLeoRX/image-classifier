package com.tekleo.image_classifier.dataset;

/**
 * Exception to be used when initializing CIFAR dataset
 *
 * @author Leo Ertuna
 * @since 31.03.2018 22:07
 */
public class CifarDatasetInitException extends Exception {
    public CifarDatasetInitException() {

    }

    public CifarDatasetInitException(String message) {
        super(message);
    }

    public CifarDatasetInitException(String message, Throwable cause) {
        super(message, cause);
    }

    public CifarDatasetInitException(Throwable cause) {
        super(cause);
    }

    public CifarDatasetInitException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}