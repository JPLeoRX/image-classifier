package com.tekleo.image_classifier.neural_network.config;

/**
 * @author Leo Ertuna
 * @since 01.04.2018 17:47
 */
public class ConfigRuntimeException extends RuntimeException {
    public ConfigRuntimeException() {

    }

    public ConfigRuntimeException(String message) {
        super(message);
    }

    public ConfigRuntimeException(String message, Throwable cause) {
        super(message, cause);
    }

    public ConfigRuntimeException(Throwable cause) {
        super(cause);
    }

    public ConfigRuntimeException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}