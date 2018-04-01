package com.tekleo.image_classifier.neural_network.config.loader;

/**
 * Exception to be used in {@link ConfigLoader}
 *
 * @author Leo Ertuna
 * @since 01.04.2018 18:17
 */
public class ConfigLoaderException extends Exception {
    public ConfigLoaderException() {

    }

    public ConfigLoaderException(String message) {
        super(message);
    }

    public ConfigLoaderException(String message, Throwable cause) {
        super(message, cause);
    }

    public ConfigLoaderException(Throwable cause) {
        super(cause);
    }

    public ConfigLoaderException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}