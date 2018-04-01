package com.tekleo.image_classifier.neural_network.config;

/**
 * @author Leo Ertuna
 * @since 01.04.2018 18:22
 */
public class ConfigException extends Exception {
    public ConfigException() {

    }

    public ConfigException(String message) {
        super(message);
    }

    public ConfigException(String message, Throwable cause) {
        super(message, cause);
    }

    public ConfigException(Throwable cause) {
        super(cause);
    }

    public ConfigException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}