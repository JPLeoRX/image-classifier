package com.tekleo.image_classifier;

import com.tekleo.image_classifier.dataset.CifarDataset;
import com.tekleo.image_classifier.neural_network.CifarModel;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;

/**
 * Main class to run
 * @author Leo Ertuna
 * @since 01.04.2018 01:07
 */
public class Main {
    public static void main(String[] args) throws Exception  {
        CifarDataset cifarDataset = new CifarDataset();
        CifarModel cifarModel = new CifarModel(cifarDataset.getTrainDataSetIterator(), cifarDataset.getTestDataSetIterator());
        cifarModel.train();
        //cifarModel.load();
        cifarModel.test();
        cifarModel.save();
    }
}