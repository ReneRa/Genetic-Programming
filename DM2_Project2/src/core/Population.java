package core;

import java.io.Serializable;
import java.util.ArrayList;

public class Population implements Serializable {
	private static final long serialVersionUID = 7L;
	protected ArrayList<Individual> individuals;

	public Population() {
		individuals = new ArrayList<Individual>();
	}

	public Individual getBest() {
		return individuals.get(getBestIndex());
	}

	public int getBestIndex() {
		int bestIndex = 0;
		double bestTrainingError = individuals.get(bestIndex).getTrainingError();
		for (int i = 1; i < individuals.size(); i++) {
			if (individuals.get(i).getTrainingError() < bestTrainingError) {
				bestTrainingError = individuals.get(i).getTrainingError();
				bestIndex = i;
			}
		}
		return bestIndex;

		// int bestIndex = 0;
		// double bestTrainingError = (1.2 *
		// individuals.get(bestIndex).getTrainingError())
		// + (0.1 * individuals.get(bestIndex).getAbsErrorDiff());
		// for (int i = 1; i < individuals.size(); i++) {
		// double currentError = (1.2 * individuals.get(i).getTrainingError())
		// + (0.1 * individuals.get(i).getAbsErrorDiff());
		// double actualError = individuals.get(i).getTrainingError();
		// if (currentError < bestTrainingError) {
		// bestTrainingError = actualError;
		// bestIndex = i;
		// }
		// }
		// return bestIndex;
	}

	// Returns the best N to be used as a seed population
	public ArrayList<Individual> getBestIndividuals(int n) {
		ArrayList<Individual> bestIndividuals = new ArrayList<Individual>();
		ArrayList<Individual> tempPopulation = individuals;
		for (int i = 0; i < n; i++) {
			int bestIndex = 0;
			double bestTrainingError = tempPopulation.get(bestIndex).getTrainingError();
			for (int j = 1; j < tempPopulation.size(); j++) {
				if (tempPopulation.get(j).getTrainingError() < bestTrainingError) {
					bestTrainingError = tempPopulation.get(j).getTrainingError();
					bestIndex = j;
				}
			}
			bestIndividuals.add(tempPopulation.get(bestIndex));
			tempPopulation.remove(bestIndex);
		}
		return bestIndividuals;
	}

	public void addIndividual(Individual individual) {
		individuals.add(individual);
	}

	public void removeIndividual(int index) {
		individuals.remove(index);
	}

	public int getSize() {
		return individuals.size();
	}

	public Individual getIndividual(int index) {
		return individuals.get(index);
	}
}
