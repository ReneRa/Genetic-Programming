package utils;

import core.Individual;
import core.Main;

public class FinalEvaluation {

	public static void main(String[] args) {
		// read individual
		Individual finalIndividual = (Individual) Utils.readObjectFromFile("final_individual.obj");

		// evaluate individual on final data
		// String evaluationData = "dataset.txt";
		String evaluationData = "dataset_training.txt";
		double[][] data = Main.readData(evaluationData);
		double[] outputs = finalIndividual.evaluate(data);
		double finalError = Utils.calculateRMSE(data, outputs);
		System.out.println(finalError);
	}
}
