package com.tekleo.image_classifier.neural_network.config.loader;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.charset.Charset;

/**
 * Helper class with static method to load config objects based on their JSON files
 *
 * @author Leo Ertuna
 * @since 01.04.2018 18:21
 */
public class ConfigLoader {
    private static final GsonBuilder GSON_BUILDER = new GsonBuilder();
    private static final Gson GSON = GSON_BUILDER.create();

    public static <E> E load(String pathToJsonFileInResourcesDirectory, Class<E> classOfE) {
        try {
            return getObjectFromJson(getStringFromFile(getFile(pathToJsonFileInResourcesDirectory)), classOfE);
        } catch (ConfigLoaderException e) {
            throw new ConfigLoaderRuntimeException("Failed to load: ", e);
        }
    }

    // Static helpers
    //------------------------------------------------------------------------------------------------------------------
    /**
     * Load a file from resource folder
     * @param pathInResourcesDirectory
     * @return
     */
    private static File getFile(String pathInResourcesDirectory) throws ConfigLoaderException {
        try {
            return new File(ConfigLoader.class.getClassLoader().getResource(pathInResourcesDirectory).toURI());
        } catch (Exception e) {
            throw new ConfigLoaderException("Exception: ", e);
        }
    }

    /**
     * Read file contents into a string
     * @param file
     * @return
     */
    private static String getStringFromFile(File file) throws ConfigLoaderException {
        try {
            return FileUtils.readFileToString(file, Charset.forName("UTF-8"));
        } catch (Exception e) {
            throw new ConfigLoaderException("Exception: ", e);
        }
    }

    /**
     * Build an object from JSON string
     * @param jsonString
     * @param classOfE
     * @param <E>
     * @return
     * @throws ConfigLoaderException
     */
    private static <E> E getObjectFromJson(String jsonString, Class<E> classOfE) throws ConfigLoaderException {
        try {
            return GSON.fromJson(jsonString, classOfE);
        } catch (Exception e) {
            throw new ConfigLoaderException("Exception: ", e);
        }
    }
    //------------------------------------------------------------------------------------------------------------------
}