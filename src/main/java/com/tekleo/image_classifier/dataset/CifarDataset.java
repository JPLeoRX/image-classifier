package com.tekleo.image_classifier.dataset;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Random;

/**
 * An easy to use image data pipeline that reads all images from resource folder and prepares them for machine learning
 * Some image/setup properties are pre-defined as static fields TODO move them into some sort of external config file
 *
 * @author Leo Ertuna
 * @since 01.04.2018 00:57
 */
public class CifarDataset {
    private static final int IMAGE_HEIGHT = 32;
    private static final int IMAGE_WIDTH = 32;
    private static final int IMAGE_CHANNELS = 3;
    private static final int IMAGE_CLASSES = 10;
    private static final int SEED = 666;
    private static final String RESOURCE_DIR_NAME = "cifar";
    private static final String[] ALLOWED_FORMATS = {"png"};
    private static final int TRAIN_SPLIT_PERCENTAGE = 80;
    private static final int TEST_SPLIT_PERCENTAGE = 20;
    private static final int TRAIN_SPLIT_INDEX = 0;
    private static final int TEST_SPLIT_INDEX = 1;
    private static final int BATCH_SIZE = 240;
    private static final int LABEL_INDEX = 1;
    private static final double NORM_MIN = 0;
    private static final double NORM_MAX = 1;

    // Internal components of image data pipeline
    private Random randomNumbersGenerator;
    private URL resourceDirectoryURL;
    private URI resourceDirectoryURI;
    private File resourceDirectoryFile;
    private FileSplit filesInResourceDirectoryFileSplit;
    private ParentPathLabelGenerator labelGenerator;
    private BalancedPathFilter pathFilter;
    private InputSplit[] filesInResourceDirectoryInputSplit;
    private InputSplit trainDataInputSplit;
    private InputSplit testDataInputSplit;
    private ImageRecordReader trainDataImageRecordReader;
    private ImageRecordReader testDataImageRecordReader;
    private RecordReaderDataSetIterator trainDataSetIterator;
    private RecordReaderDataSetIterator testDataSetIterator;
    private ImagePreProcessingScaler imagePreProcessingScaler;

    public CifarDataset() {
        this.init();
    }

    private void init() {
        try {
            // Init random numbers generator for splitting between train/test
            randomNumbersGenerator = initRandomNumbersGenerator(SEED);

            // Init URL, URI and File, then check that the directory is good to go
            resourceDirectoryURL = initResourceDirectoryURL(RESOURCE_DIR_NAME);
            resourceDirectoryURI = initResourceDirectoryURI(resourceDirectoryURL);
            resourceDirectoryFile = initResourceDirectoryFile(resourceDirectoryURI);
            initCheckResourceDirectory(resourceDirectoryFile);

            // Init file split, label maker, path filter
            filesInResourceDirectoryFileSplit = initFilesInResourceDirectoryFileSplit(resourceDirectoryFile, ALLOWED_FORMATS, randomNumbersGenerator);
            labelGenerator = initLabelGenerator();
            pathFilter = initPathFilter(randomNumbersGenerator, ALLOWED_FORMATS, labelGenerator);

            // Init input split, split into train/test
            filesInResourceDirectoryInputSplit = initFilesInResourceDirectoryInputSplit(filesInResourceDirectoryFileSplit, pathFilter, TRAIN_SPLIT_PERCENTAGE, TEST_SPLIT_PERCENTAGE);
            trainDataInputSplit = filesInResourceDirectoryInputSplit[TRAIN_SPLIT_INDEX];
            testDataInputSplit = filesInResourceDirectoryInputSplit[TEST_SPLIT_INDEX];

            // Init image record readers
            trainDataImageRecordReader = initImageRecordReader(trainDataInputSplit, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelGenerator);
            testDataImageRecordReader = initImageRecordReader(testDataInputSplit, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelGenerator);

            // Init data set iterators
            trainDataSetIterator = initRecordReaderDataSetIterator(trainDataImageRecordReader, BATCH_SIZE, LABEL_INDEX, IMAGE_CLASSES);
            testDataSetIterator = initRecordReaderDataSetIterator(testDataImageRecordReader, BATCH_SIZE, LABEL_INDEX, IMAGE_CLASSES);

            // Init data normalization, fit data into it, save this preprocessor into data set iterators
            imagePreProcessingScaler = initImagePreProcessingScaler(NORM_MIN, NORM_MAX);
            imagePreProcessingScaler.fit(trainDataSetIterator);
            imagePreProcessingScaler.fit(testDataSetIterator);
            trainDataSetIterator.setPreProcessor(imagePreProcessingScaler);
            testDataSetIterator.setPreProcessor(imagePreProcessingScaler);
        }

        catch (CifarDatasetInitException e) {
            throw new RuntimeException("Failed to initialize due to: ", e);
        }
    }



    // Getters
    //------------------------------------------------------------------------------------------------------------------
    public DataSetIterator getTrainDataSetIterator() {
        return trainDataSetIterator;
    }

    public DataSetIterator getTestDataSetIterator() {
        return testDataSetIterator;
    }
    //------------------------------------------------------------------------------------------------------------------



    // Static helpers - Initialization
    //------------------------------------------------------------------------------------------------------------------
    private static Random initRandomNumbersGenerator(final int seed) throws CifarDatasetInitException {
        try {
            return new Random(seed);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static URL initResourceDirectoryURL(final String pathInResourcesDirectory) throws CifarDatasetInitException {
        try {
            return CifarDataset.class.getClassLoader().getResource(pathInResourcesDirectory);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static URI initResourceDirectoryURI(final URL resourceDirectoryURL) throws CifarDatasetInitException {
        try {
            return resourceDirectoryURL.toURI();
        } catch (URISyntaxException e) {
            throw new CifarDatasetInitException("URI Syntax Exception: ", e);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static File initResourceDirectoryFile(final URI resourceDirectoryURI) throws CifarDatasetInitException {
        try {
            return new File(resourceDirectoryURI);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static void initCheckResourceDirectory(final File resourceDirectoryFile) throws CifarDatasetInitException {
        if (!resourceDirectoryFile.exists())
            throw new CifarDatasetInitException(resourceDirectoryFile.getPath() + " doesn't exist");

        if (!resourceDirectoryFile.isDirectory())
            throw new CifarDatasetInitException(resourceDirectoryFile.getPath() + " is not a directory");
    }

    private static FileSplit initFilesInResourceDirectoryFileSplit(final File resourceDirectoryFile, final String[] allowedFormats, final Random randomNumbersGenerator) throws CifarDatasetInitException {
        try {
            return new FileSplit(resourceDirectoryFile, allowedFormats, randomNumbersGenerator);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static ParentPathLabelGenerator initLabelGenerator() throws CifarDatasetInitException {
        try {
            return new ParentPathLabelGenerator();
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static BalancedPathFilter initPathFilter(final Random randomNumbersGenerator, final String[] allowedFormats, final ParentPathLabelGenerator labelGenerator) throws CifarDatasetInitException {
        try {
            return new BalancedPathFilter(randomNumbersGenerator, allowedFormats, labelGenerator);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static InputSplit[] initFilesInResourceDirectoryInputSplit(final FileSplit filesInResourceDirectoryFileSplit, final BalancedPathFilter pathFilter, final int trainPercentage, final int testPercentage) throws CifarDatasetInitException {
        try {
            return filesInResourceDirectoryFileSplit.sample(pathFilter, trainPercentage, testPercentage);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static ImageRecordReader initImageRecordReader(final InputSplit inputSplit, final int height, final int width, final int channels, final ParentPathLabelGenerator labelGenerator) throws CifarDatasetInitException {
        try {
            ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, labelGenerator);
            imageRecordReader.initialize(inputSplit);
            return imageRecordReader;
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static RecordReaderDataSetIterator initRecordReaderDataSetIterator(final ImageRecordReader imageRecordReader, final int batchSize, final int labelIndex, final int numPossibleLabels) throws CifarDatasetInitException {
        try {
            return new RecordReaderDataSetIterator(imageRecordReader, batchSize, labelIndex, numPossibleLabels);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }

    private static ImagePreProcessingScaler initImagePreProcessingScaler(final double min, final double max) throws CifarDatasetInitException {
        try {
            return new ImagePreProcessingScaler(min, max);
        } catch (Exception e) {
            throw new CifarDatasetInitException("Exception: ", e);
        }
    }
    //------------------------------------------------------------------------------------------------------------------
}