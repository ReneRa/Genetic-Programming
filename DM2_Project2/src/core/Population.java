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
