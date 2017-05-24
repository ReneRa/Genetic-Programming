package weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import utils.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class WekaRun {

	public static final boolean SHUFFLE_AND_SPLIT = true;
	// public static final boolean SHUFFLE_AND_SPLIT = false;

	public static final boolean REGRESSION = true;
	// public static final boolean REGRESSION = false;

	public static Instances trainingData, unseenData;

	public static void main(String[] args) throws Exception {

		ArrayList<String> datasets = getDatasets();
		ArrayList<Classifier> classifiers = getClassifiers();

		for (String dataset : datasets) {
			System.out.printf("\n\t\t##### Dataset: %s #####\n\n", dataset);
			// load training and unseen data
			loadData(dataset);
			// run classifiers
			runClassifiers(classifiers);
		}
	}

	public static ArrayList<Classifier> getClassifiers() throws Exception {
		ArrayList<Classifier> classifiers = new ArrayList<Classifier>();
		if (REGRESSION) {
			classifiers.add(new RandomForest());
			classifiers.add(new SMOreg());

			// MLP with different parameters
			MultilayerPerceptron mlpDifferentParameter = new MultilayerPerceptron();
			String[] options = { "-L", "0.75", "-N", "200", "-H", "5" };
			mlpDifferentParameter.setOptions(options);
			classifiers.add(mlpDifferentParameter);

		} else {
			classifiers.add(new RandomForest());
			classifiers.add(new SMO());

			// MLP with different parameters
			MultilayerPerceptron mlpDifferentParameter = new MultilayerPerceptron();
			String[] options = { "-L", "0.75", "-N", "200", "-H", "5" };
			mlpDifferentParameter.setOptions(options);
			classifiers.add(mlpDifferentParameter);

			classifiers.add(new J48());
		}
		return classifiers;
	}

	public static ArrayList<String> getDatasets() {
		ArrayList<String> datasets = new ArrayList<String>();
		if (REGRESSION) {
			datasets.add("dataset");
			// datasets.add("bio");
			// datasets.add("ppb");
			// datasets.add("ld50");
		} else {
			datasets.add("iris");
		}
		return datasets;
	}

	public static void runClassifiers(ArrayList<Classifier> classifiers) {
		try {
			System.out.println("Classifier\n\t\t\tTraining data\tUnseen data\n");
			for (int i = 0; i < classifiers.size(); i++) {
				Classifier classifier = classifiers.get(i);
				Evaluation[] results = classify(classifier);
				double unseenDataError = calculateError(results[0]);
				double trainingDataError = calculateError(results[1]);
				System.out.printf("%s\n\t\t\t%.2f\t\t%.2f\n", classifier.getClass().getSimpleName(), trainingDataError,
						unseenDataError);
				// uncomment to show the classifier details
				// System.out.println(classifier);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static Evaluation[] classify(Classifier classifier) throws Exception {
		Evaluation[] evaluation = new Evaluation[2];
		evaluation[0] = new Evaluation(trainingData);
		evaluation[1] = new Evaluation(trainingData);
		classifier.buildClassifier(trainingData);
		evaluation[0].evaluateModel(classifier, unseenData);
		evaluation[1].evaluateModel(classifier, trainingData);
		return evaluation;
	}

	public static double calculateError(Evaluation evaluation) {
		if (REGRESSION) {
			return calculateRMSE(evaluation);
		} else {
			return calculateAccuracy(evaluation);
		}
	}

	public static double calculateRMSE(Evaluation evaluation) {
		double sum = 0;
		ArrayList<Prediction> predictions = evaluation.predictions();
		for (Prediction prediction : predictions) {
			sum += Math.pow(prediction.predicted() - prediction.actual(), 2);
		}
		return Math.sqrt(sum / evaluation.numInstances());
	}

	public static double calculateAccuracy(Evaluation evaluation) {
		double correct = 0;
		ArrayList<Prediction> predictions = evaluation.predictions();
		for (Prediction prediction : predictions) {
			if (prediction.predicted() == prediction.actual()) {
				correct++;
			}
		}
		return 100 * correct / evaluation.numInstances();
	}

	public static void loadData(String dataFilename) {

		if (SHUFFLE_AND_SPLIT) {
			Instances allData = readArffData(dataFilename + ".arff");
			List<Integer> instances = Utils.shuffleInstances(allData.size());
			int trainingInstances = (int) Math.floor(0.7 * allData.size());
			int unseenInstances = (int) Math.ceil(0.3 * allData.size());

			trainingData = new Instances(allData, trainingInstances);
			unseenData = new Instances(allData, unseenInstances);

			for (int i = 0; i < trainingInstances; i++) {
				trainingData.add(allData.get(instances.get(i)));
			}

			for (int i = 0; i < unseenInstances; i++) {
				unseenData.add(allData.get(instances.get(trainingInstances + i)));
			}
		} else {
			trainingData = readArffData(dataFilename + "_training.arff");
			unseenData = readArffData(dataFilename + "_unseen.arff");
		}
	}

	public static Instances readArffData(String filename) {
		BufferedReader datafile = readFile(filename);
		Instances data = null;
		try {
			data = new Instances(datafile);
		} catch (IOException e) {
			System.err.println("Error loading data: " + e.getLocalizedMessage());
		}
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	protected static BufferedReader readFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}
}
