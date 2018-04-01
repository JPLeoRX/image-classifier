package com.tekleo.image_classifier.neural_network.config.loader;

/**
 * Exception to be used in {@link ConfigLoader}
 *
 * @author Leo Ertuna
 * @since 01.04.2018 18:17
 */
public class ConfigLoaderRuntimeException extends RuntimeException {
    public ConfigLoaderRuntimeException() {

    }

    public ConfigLoaderRuntimeException(String message) {
        super(message);
    }

    public ConfigLoaderRuntimeException(String message, Throwable cause) {
        super(message, cause);
    }

    public ConfigLoaderRuntimeException(Throwable cause) {
        super(cause);
    }

    public ConfigLoaderRuntimeException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}