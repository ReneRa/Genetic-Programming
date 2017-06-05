package core;

import programElements.Addition;
import programElements.Constant;
import programElements.LogisticFunction;
import programElements.Multiplication;
import programElements.Subtraction;
import utils.Utils;

public class GsgpRun extends GpRun {
	private static final long serialVersionUID = 7L;
	protected double mutationStep;
	protected boolean boundedMutation;
	protected boolean buildIndividuals;
	protected boolean seedInitialization;

	public GsgpRun(Data data, boolean interleavedSampling) {
		super(data, interleavedSampling);
	}

	@Override
	protected void initialize() {
		super.initialize();
		applyDepthLimit = false;
		mutationStep = 1.0;
		boundedMutation = false;
		// boundedMutation = true;
		// buildIndividuals = true;
		buildIndividuals = false;
	}

	@Override
	protected Individual applyStandardCrossover(Individual p1, Individual p2) {
		if (buildIndividuals) {
			return buildCrossoverIndividual(p1, p2);
		} else {
			return buildCrossoverSemantics(p1, p2);
		}
	}

	protected Individual buildCrossoverIndividual(Individual p1, Individual p2) {
		Individual offspring = new Individual();
		offspring.addProgramElement(new Addition());
		offspring.addProgramElement(new Multiplication());
		// copy first parent to offspring
		for (int i = 0; i < p1.getSize(); i++) {
			offspring.addProgramElement(p1.getProgramElementAtIndex(i));
		}
		// create a random tree
		int maximumInitialDepth = 6;
		Individual randomTree = grow(maximumInitialDepth);
		offspring.addProgramElement(new LogisticFunction());
		// copy random tree to offspring
		for (int i = 0; i < randomTree.getSize(); i++) {
			offspring.addProgramElement(randomTree.getProgramElementAtIndex(i));
		}
		offspring.addProgramElement(new Multiplication());
		offspring.addProgramElement(new Subtraction());
		offspring.addProgramElement(new Constant(1.0));
		offspring.addProgramElement(new LogisticFunction());
		// copy random tree to offspring
		for (int i = 0; i < randomTree.getSize(); i++) {
			offspring.addProgramElement(randomTree.getProgramElementAtIndex(i));
		}
		// copy second parent to offspring
		for (int i = 0; i < p2.getSize(); i++) {
			offspring.addProgramElement(p2.getProgramElementAtIndex(i));
		}
		offspring.calculateDepth();
		return offspring;
	}

	protected Individual buildCrossoverSemantics(Individual p1, Individual p2) {
		Individual offspring = new Individual();
		// create a random tree and evaluate it
		int maximumInitialDepth = 6;
		Individual randomTree = grow(maximumInitialDepth);
		randomTree.evaluate(data);
		// build training data semantics
		double[] parent1TrainingSemantics = p1.getTrainingDataOutputs();
		double[] parent2TrainingSemantics = p2.getTrainingDataOutputs();
		double[] randomTreeTrainingSemantics = randomTree.getTrainingDataOutputs();
		double[] offspringTrainingSemantics = buildCrossoverOffspringSemantics(parent1TrainingSemantics,
				parent2TrainingSemantics, randomTreeTrainingSemantics);
		offspring.setTrainingDataOutputs(offspringTrainingSemantics);
		// build unseen data semantics
		double[] parent1UnseenSemantics = p1.getUnseenDataOutputs();
		double[] parent2UnseenSemantics = p2.getUnseenDataOutputs();
		double[] randomTreeUnseenSemantics = randomTree.getUnseenDataOutputs();
		double[] offspringUnseenSemantics = buildCrossoverOffspringSemantics(parent1UnseenSemantics,
				parent2UnseenSemantics, randomTreeUnseenSemantics);
		offspring.setUnseenDataOutputs(offspringUnseenSemantics);
		// calculate size and depth
		offspring.setSizeOverride(true);
		offspring.setComputedSize(calculateCrossoverOffspringSize(p1, p2, randomTree));
		offspring.setDepth(calculateCrossoverOffspringDepth(p1, p2, randomTree));
		return offspring;
	}

	protected double[] buildCrossoverOffspringSemantics(double[] parent1Semantics, double[] parent2Semantics,
			double[] randomTreeSemantics) {
		double[] offspringSemantics = new double[parent1Semantics.length];
		for (int i = 0; i < offspringSemantics.length; i++) {
			double randomTreeValue = Utils.logisticFunction(randomTreeSemantics[i]);
			offspringSemantics[i] = (parent1Semantics[i] * randomTreeValue)
					+ ((1.0 - randomTreeValue) * parent2Semantics[i]);
		}
		return offspringSemantics;
	}

	protected int calculateCrossoverOffspringSize(Individual p1, Individual p2, Individual randomTree) {
		return p1.getSize() + p2.getSize() + randomTree.getSize() * 2 + 5;
	}

	protected int calculateCrossoverOffspringDepth(Individual p1, Individual p2, Individual randomTree) {
		int largestParentDepth = Math.max(p1.getDepth(), p2.getDepth());
		// "+ 1" because of the bounding function
		return Math.max(largestParentDepth + 2, randomTree.getDepth() + 3 + 1);
	}

	@Override
	protected Individual applyStandardMutation(Individual p) {
		if (buildIndividuals) {
			return buildMutationIndividual(p);
		} else {
			return buildMutationSemantics(p);
		}
	}

	protected Individual buildMutationIndividual(Individual p) {
		Individual offspring = new Individual();
		offspring.addProgramElement(new Addition());
		// copy parent to offspring
		for (int i = 0; i < p.getSize(); i++) {
			offspring.addProgramElement(p.getProgramElementAtIndex(i));
		}
		offspring.addProgramElement(new Multiplication());
		offspring.addProgramElement(new Constant(mutationStep));
		offspring.addProgramElement(new Subtraction());
		// create 2 random trees
		int maximumInitialDepth = 6;
		Individual randomTree1 = grow(maximumInitialDepth);
		Individual randomTree2 = grow(maximumInitialDepth);
		if (boundedMutation) {
			offspring.addProgramElement(new LogisticFunction());
		}
		// copy random tree 1 to offspring
		for (int i = 0; i < randomTree1.getSize(); i++) {
			offspring.addProgramElement(randomTree1.getProgramElementAtIndex(i));
		}
		if (boundedMutation) {
			offspring.addProgramElement(new LogisticFunction());
		}
		// copy random tree 2 to offspring
		for (int i = 0; i < randomTree2.getSize(); i++) {
			offspring.addProgramElement(randomTree2.getProgramElementAtIndex(i));
		}
		offspring.calculateDepth();
		return offspring;
	}

	protected Individual buildMutationSemantics(Individual p) {
		Individual offspring = new Individual();
		// create 2 random trees and evaluate them
		int maximumInitialDepth = 6;
		Individual randomTree1 = grow(maximumInitialDepth);
		Individual randomTree2 = grow(maximumInitialDepth);
		randomTree1.evaluate(data);
		randomTree2.evaluate(data);
		// build training data semantics
		double[] parentTrainingSemantics = p.getTrainingDataOutputs();
		double[] randomTree1TrainingSemantics = randomTree1.getTrainingDataOutputs();
		double[] randomTree2TrainingSemantics = randomTree2.getTrainingDataOutputs();
		double[] offspringTrainingSemantics = buildMutationOffspringSemantics(parentTrainingSemantics,
				randomTree1TrainingSemantics, randomTree2TrainingSemantics);
		offspring.setTrainingDataOutputs(offspringTrainingSemantics);
		// build unseen data semantics
		double[] parentUnseenSemantics = p.getUnseenDataOutputs();
		double[] randomTree1UnseenSemantics = randomTree1.getUnseenDataOutputs();
		double[] randomTree2UnseenSemantics = randomTree2.getUnseenDataOutputs();
		double[] offspringUnseenSemantics = buildMutationOffspringSemantics(parentUnseenSemantics,
				randomTree1UnseenSemantics, randomTree2UnseenSemantics);
		offspring.setUnseenDataOutputs(offspringUnseenSemantics);
		// calculate size and depth
		offspring.setSizeOverride(true);
		offspring.setComputedSize(calculateMutationOffspringSize(p, randomTree1, randomTree2));
		offspring.setDepth(calculateMutationOffspringDepth(p, randomTree1, randomTree2));
		return offspring;
	}

	protected double[] buildMutationOffspringSemantics(double[] parentSemantics, double[] randomTree1Semantics,
			double[] randomTree2Semantics) {
		double[] offspringSemantics = new double[parentSemantics.length];
		for (int i = 0; i < offspringSemantics.length; i++) {
			double value1 = randomTree1Semantics[i];
			double value2 = randomTree2Semantics[i];
			if (boundedMutation) {
				value1 = Utils.logisticFunction(value1);
				value2 = Utils.logisticFunction(value2);
			}
			offspringSemantics[i] = parentSemantics[i] + (mutationStep * (value1 - value2));
		}
		return offspringSemantics;
	}

	protected int calculateMutationOffspringSize(Individual parent, Individual randomTree1, Individual randomTree2) {
		return parent.getSize() + randomTree1.getSize() + randomTree2.getSize() + 4;
	}

	protected int calculateMutationOffspringDepth(Individual parent, Individual randomTree1, Individual randomTree2) {
		int largestRandomTreeDepth = Math.max(randomTree1.getDepth(), randomTree2.getDepth());
		return Math.max(largestRandomTreeDepth + 3, parent.getDepth() + 1);
	}

	// ##### get's and set's from here on #####
	public double getMutationStep() {
		return mutationStep;
	}

	public void setMutationStep(double mutationStep) {
		this.mutationStep = mutationStep;
	}

	public void setBoundedMutation(boolean boundedMutation) {
		this.boundedMutation = boundedMutation;
	}

	public void setBuildIndividuals(boolean buildIndividuals) {
		this.buildIndividuals = buildIndividuals;
	}
}
