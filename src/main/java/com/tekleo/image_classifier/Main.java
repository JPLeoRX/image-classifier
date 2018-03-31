package com.tekleo.image_classifier;

import com.tekleo.image_classifier.dataset.CifarDataset;
import com.tekleo.image_classifier.neural_network.CifarModel;

/**
 * Main class to run
 * @author Leo Ertuna
 * @since 01.04.2018 01:07
 */
public class Main {
    public static void main(String[] args)  {
        CifarDataset cifarDataset = new CifarDataset();
        CifarModel cifarModel = new CifarModel(cifarDataset.getTrainDataSetIterator(), cifarDataset.getTestDataSetIterator());
        cifarModel.train();
    }
}