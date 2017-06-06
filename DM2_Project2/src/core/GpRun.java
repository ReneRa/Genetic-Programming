package core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import programElements.Addition;
import programElements.Constant;
import programElements.InputVariable;
import programElements.Multiplication;
import programElements.Operator;
import programElements.ProgramElement;
import programElements.ProtectedDivision;
import programElements.Subtraction;
import programElements.Terminal;
import programElements.LogisticFunction;
import programElements.Median;
import programElements.randMult;
import programElements.randDiv;
import programElements.Max;
import programElements.Min;
import programElements.Power;
import programElements.Root;
import programElements.aSquare;
import programElements.SquareRoot;
import programElements.Log;
import programElements.Log10;
import utils.Utils;

public class GpRun implements Serializable {
	private static final long serialVersionUID = 7L;

	// ##### parameters #####
	protected Data data;
	protected double[][] originalTrainingData;
	protected double[][] originalUnseenData;
	protected double[][] originalKFoldTrainingData;
	protected double[][] previousTrainingData;
	protected double[][] previousUnseenData;
	protected ArrayList<ProgramElement> functionSet, otherOperator, terminalSet, fullSet;
	protected int populationSize;
	protected boolean applyDepthLimit;
	protected int maximumDepth;
	protected double crossoverProbability;
	protected double mutationProbability;
	protected boolean printAtEachGeneration;
	protected double probToGrowOperator;
	protected double probOtherOperator;
	protected int mutationOperator;
	protected int maxRunsWithoutImprovements;

	// ##### state #####
	protected Random randomGenerator;
	protected int currentGeneration;
	protected Population population;
	protected Individual currentBest;
	protected Individual globalBest;
	protected int constantsLength;
	protected double maxNumberOfMutations;
	protected int runsWithoutImprovement;
	protected boolean interleavedSampling;
	protected double chooseSingleSubSampleProbability = 0.75;
	protected double subSampleSize = 0.35;

	public GpRun(Data data, Data kFoldData, boolean interleavedSampling) {
		this.data = data;
		this.originalTrainingData = data.getTrainingData();
		this.originalUnseenData = data.getUnseenData();
		this.originalKFoldTrainingData = kFoldData.getTrainingData();
		this.interleavedSampling = interleavedSampling;
		initialize();
	}

	protected void initialize() {
		// adds all the functions to the function set
		functionSet = new ArrayList<ProgramElement>();
		functionSet.add(new Addition());
		functionSet.add(new Subtraction());
		functionSet.add(new Multiplication());
		functionSet.add(new ProtectedDivision());
		
		otherOperator = new ArrayList<ProgramElement>();
		otherOperator.add(new Min());
		otherOperator.add(new Max());
		otherOperator.add(new Median());
		otherOperator.add(new Root());
		otherOperator.add(new Power());
		otherOperator.add(new aSquare());
		otherOperator.add(new SquareRoot());
		otherOperator.add(new Log());
		otherOperator.add(new Log10());
		
		if (this.interleavedSampling == true) {
			data.trainingData = getSubSample(originalTrainingData, chooseSingleSubSampleProbability);
		}
		// adds all the constants to the terminal set
		terminalSet = new ArrayList<ProgramElement>();
		double[] constants = { -1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1 };
		constantsLength = constants.length;
		for (int i = 0; i < constantsLength; i++) {
			terminalSet.add(new Constant(constants[i]));
		}
		// adds all the input variables to the terminal set
		for (int i = 0; i < data.getDimensionality(); i++) {
			terminalSet.add(new InputVariable(i));
		}
		// creates the set which contains all the program elements
		fullSet = new ArrayList<ProgramElement>();
		for (ProgramElement programElement : functionSet) {
			fullSet.add(programElement);
		}
		for (ProgramElement programElement : terminalSet) {
			fullSet.add(programElement);
		}
		for (ProgramElement programElement : otherOperator) {
			fullSet.add(programElement);
		}
		
		populationSize = 100;
		applyDepthLimit = true;
		maximumDepth = 17;
		crossoverProbability = 0.90;
		mutationProbability = 0.1;
		probToGrowOperator = 0.5;
		probOtherOperator = 0.8;
		printAtEachGeneration = true;
		mutationOperator = 1;
		maxNumberOfMutations = 0.05;
		runsWithoutImprovement = 0;
		// you can turn it off by changing % of generations to 1
		maxRunsWithoutImprovements = Math.round(Main.NUMBER_OF_GENERATIONS * 1);
		currentGeneration = 0;
		// initialize and evaluate po..pulation
		rampedHalfAndHalfInitialization();
		for (int i = 0; i < populationSize; i++) {
			if (Main.kFold == true & randomGenerator.nextDouble() < Main.kFoldProbability) {
				kFoldEvaluation(originalKFoldTrainingData, population.getIndividual(i));
			} else {
				population.getIndividual(i).evaluate(data);
			}
		}
		updateCurrentBest();
		globalBest = getCurrentBest();
		printState();
		storeValues();
		currentGeneration++;
	}

	// Validates an individual using k fold cross validation
	private void kFoldEvaluation(double[][] originalTrainingData, Individual individual) {
		int partitionLength = originalTrainingData.length / Main.k;
		int trainingSetSize = partitionLength * (Main.k - 1);
		double averageTrainingError = 0;
		double averageUnseenError = 0;

		for (int k = 0; k < Main.k; k++) {
			double[][] trainingSet = new double[trainingSetSize][];
			double[][] validationSet = new double[partitionLength][];

			for (int i = 0 + (k * partitionLength); i < partitionLength + (partitionLength * k); i++) {
				validationSet[i - (k * partitionLength)] = originalTrainingData[i];
			}

			// Load training data that is located before the validation set
			for (int i = 0; i < k * partitionLength; i++) {
				trainingSet[i] = originalTrainingData[i];
			}
			// Load training data that is located after the validation set
			for (int i = (k + 1) * partitionLength; i < originalTrainingData.length; i++) {
				trainingSet[i - partitionLength] = originalTrainingData[i];

			}

			data.trainingData = trainingSet;
			data.unseenData = validationSet;

			individual.evaluate(data);
			averageTrainingError += individual.getTrainingError();
			averageUnseenError += individual.getUnseenError();
		}
		averageTrainingError = averageTrainingError / Main.k;
		averageUnseenError = averageUnseenError / Main.k;
		individual.trainingError = averageTrainingError;
		individual.unseenError = averageUnseenError;
	}

	// Returns a sub selection of the training data, used for cross validation
	private double[][] getSubSample(double[][] originalTrainingData, double chooseSingleSubSampleProbability) {
		if (randomGenerator.nextDouble() > chooseSingleSubSampleProbability) {
			return originalTrainingData;
		} else {
			double[][] trainingSubSample;
			List<Integer> instances = Utils.shuffleInstances(originalTrainingData.length);
			int subSampleInstances = (int) Math.ceil(subSampleSize * originalTrainingData.length);
			trainingSubSample = new double[subSampleInstances][];
			for (int i = 0; i < subSampleInstances; i++) {
				trainingSubSample[i] = originalTrainingData[instances.get(i)];
			}
			return trainingSubSample;
		}
	}

	// Store values so they can be written to a file later for plotting purposes
	private void storeValues() {
		Main.trainingErrors[Main.CURRENT_RUN][currentGeneration] = currentBest.getTrainingError();
		Main.unseenErrors[Main.CURRENT_RUN][currentGeneration] = currentBest.getUnseenError();
		Main.sizes[Main.CURRENT_RUN][currentGeneration] = currentBest.getSize();
		Main.depths[Main.CURRENT_RUN][currentGeneration] = currentBest.getDepth();
	}

	protected void rampedHalfAndHalfInitialization() {
		int maximumInitialDepth = 6;
		/*
		 *
		 * depth at the root node is 0. this implies that the number of
		 *
		 * different depths is equal to the maximumInitialDepth
		 *
		 */
		int individualsPerDepth = populationSize / maximumInitialDepth;
		int remainingIndividuals = populationSize % maximumInitialDepth;
		population = new Population();
		int fullIndividuals, growIndividuals;
		for (int depth = 1; depth <= maximumInitialDepth; depth++) {
			if (depth == maximumInitialDepth) {
				fullIndividuals = (int) Math.floor((individualsPerDepth + remainingIndividuals) / 2.0);
				growIndividuals = (int) Math.ceil((individualsPerDepth + remainingIndividuals) / 2.0);
			} else {
				fullIndividuals = (int) Math.floor(individualsPerDepth / 2.0);
				growIndividuals = (int) Math.ceil(individualsPerDepth / 2.0);
			}
			for (int i = 0; i < fullIndividuals; i++) {
				population.addIndividual(full(depth));
			}
			for (int i = 0; i < growIndividuals; i++) {
				population.addIndividual(grow(depth));
			}
		}
	}

	protected Individual full(int maximumTreeDepth) {
		Individual individual = new Individual();
		fullInner(individual, 0, maximumTreeDepth);
		individual.setDepth(maximumTreeDepth);
		return individual;
	}

	protected void fullInner(Individual individual, int currentDepth, int maximumTreeDepth) {
		if (currentDepth == maximumTreeDepth) {
			ProgramElement randomTerminal = terminalSet.get(randomGenerator.nextInt(terminalSet.size()));
			individual.addProgramElement(randomTerminal);
		} else {
			if (Math.random() < probOtherOperator){
			Operator randomOperator = (Operator) functionSet.get(randomGenerator.nextInt(functionSet.size()));
			individual.addProgramElement(randomOperator);
			for (int i = 0; i < randomOperator.getArity(); i++) {
				fullInner(individual, currentDepth + 1, maximumTreeDepth);
			}
			} else {
				Operator randomOperator = (Operator) otherOperator.get(randomGenerator.nextInt(otherOperator.size()));
				individual.addProgramElement(randomOperator);
				for (int i = 0; i < randomOperator.getArity(); i++) {
					fullInner(individual, currentDepth + 1, maximumTreeDepth);
				}
			}	
		}
	}

	protected Individual grow(int maximumTreeDepth) {
		Individual individual = new Individual();
		growInner(individual, 0, maximumTreeDepth, probToGrowOperator);
		individual.calculateDepth();
		return individual;
	}

	protected void growInner(Individual individual, int currentDepth, int maximumTreeDepth, double probToGrowOperator) {
		if (currentDepth == maximumTreeDepth) {
			ProgramElement randomTerminal = terminalSet.get(randomGenerator.nextInt(terminalSet.size()));
			individual.addProgramElement(randomTerminal);
		} else {
			// equal probability of adding a terminal or an operator
			if (Math.random() < probToGrowOperator) {
				if (randomGenerator.nextDouble() < probOtherOperator + 0.2){
				Operator randomOperator = (Operator) functionSet.get(randomGenerator.nextInt(functionSet.size()));
				individual.addProgramElement(randomOperator);
				for (int i = 0; i < randomOperator.getArity(); i++) {
					growInner(individual, currentDepth + 1, maximumTreeDepth, probToGrowOperator);
				}
				} else {
					Operator randomOperator = (Operator) otherOperator.get(randomGenerator.nextInt(otherOperator.size()));
					individual.addProgramElement(randomOperator);
					for (int i = 0; i < randomOperator.getArity(); i++) {
						growInner(individual, currentDepth + 1, maximumTreeDepth, probToGrowOperator);
					}
				}
			} else {
				ProgramElement randomTerminal = terminalSet.get(randomGenerator.nextInt(terminalSet.size()));
				individual.addProgramElement(randomTerminal);
			}
		}
	}

	public Individual evolve(int numberOfGenerations) {
		// alternative stopping criteria
		boolean stopCriteria = false;
		// evolve for a given number of generations
		while (currentGeneration <= numberOfGenerations && !stopCriteria) {
			previousTrainingData = data.trainingData;
			previousUnseenData = data.unseenData;
			data.trainingData = originalTrainingData;
			data.unseenData = originalUnseenData;
			boolean useKFold = false;
			if (randomGenerator.nextDouble() < Main.kFoldProbability) {
				useKFold = true;
			}
			if (this.interleavedSampling == true) {
				data.trainingData = getSubSample(originalTrainingData, this.chooseSingleSubSampleProbability);
			}
			Population offspring = new Population();
			// generate a new offspring population
			while (offspring.getSize() < population.getSize()) {

				Individual p1, newIndividual = null, newIndividual1 = null, newIndividual2 = null;
				p1 = selectParent();
				Individual p2 = selectParent();
				// apply crossover
				if (randomGenerator.nextDouble() < 1 - crossoverProbability - mutationProbability) {
					// apply reproduction
					newIndividual = p1.deepCopy();
				}
				// DELETED THE ELSE STATEMENT
				else if (randomGenerator.nextDouble() < crossoverProbability
						/ (crossoverProbability + mutationProbability)) {
					Individual[] Individuals;
					switch (Main.selectedCrossoverMethod) {
					case "uniformCrossover":
						Individuals = uniformCrossover(p1, p2);
						newIndividual1 = Individuals[0];
						newIndividual2 = Individuals[1];
						break;
					case "singleUniformCrossover":
						newIndividual = singleUniformCrossover(p1, p2);
						break;
					case "onePointCrossover":
						Individuals = onePointCrossover(p1, p2);
						newIndividual1 = Individuals[0];
						newIndividual2 = Individuals[1];
						break;
					case "standardCrossover":
						newIndividual = applyStandardCrossover(p1, p2);
						break;
					case "adjustedStandardCrossover":
						Individuals = adjustedStandardCrossover(p1, p2);
						newIndividual1 = Individuals[0];
						newIndividual2 = Individuals[1];
						break;
					case "adjustedStandardCrossoverArray":
						Individuals = applyStandardCrossoverArray(p1, p2);
						newIndividual1 = Individuals[0];
						break;
					case "randomCrossover":
						newIndividual = randomCrossover(p1, p2);
						break;
					case "adjustedRandomCrossover":
						newIndividual = adjustedRandomCrossover(p1, p2);
						break;
					case "singleOnePointCrossover":
						newIndividual = singleOnePointCrossover(p1, p2);
						break;
					case "randomTwoOffspringsCrossover":
						Individuals = randomTwoOffspringsCrossover(p1, p2);
						newIndividual1 = Individuals[0];
						newIndividual2 = Individuals[1];
						break;
					}
				}
				// apply mutation
				// if (randomGenerator.nextDouble() < mutationProbability)
				else {
					// if (p1.getSize() < 300000 ) {
					newIndividual = applyStandardMutation(p1);
					// newIndividual = applyNodeFlipMutation(p1);
					// } else {
					// newIndividual = applyConstantMutation(p1);
					// }
				}
				if (newIndividual1 == null & newIndividual2 == null) {
					if (applyDepthLimit && newIndividual.getDepth() > maximumDepth) {
						newIndividual = p1;
					} else {
						if (Main.kFold == true & useKFold) {
							kFoldEvaluation(originalKFoldTrainingData, newIndividual);
						} else {
							newIndividual.evaluate(data);
						}
					}
					offspring.addIndividual(newIndividual);
				} else if (newIndividual1 != null & newIndividual2 == null) {
					if (applyDepthLimit && newIndividual1.getDepth() > maximumDepth) {
						newIndividual1 = p1;
					} else {
						if (Main.kFold == true & useKFold) {
							kFoldEvaluation(originalKFoldTrainingData, newIndividual1);
						} else {
							newIndividual1.evaluate(data);
						}
					}
				}

				else {
					/*
					 * add the new individual to the offspring population if its
					 * depth is not higher than the maximum (applicable only if
					 * the depth limit is enabled)
					 */
					if (applyDepthLimit && newIndividual1.getDepth() > maximumDepth) {
						newIndividual1 = p1;
					} else {
						if (Main.kFold == true & useKFold) {
							kFoldEvaluation(originalKFoldTrainingData, newIndividual1);
						} else {
							newIndividual1.evaluate(data);
						}
					}
					if (applyDepthLimit && newIndividual2.getDepth() > maximumDepth) {
						newIndividual2 = p2;
					} else {
						if (Main.kFold == true & useKFold) {
							kFoldEvaluation(originalKFoldTrainingData, newIndividual2);
						} else {
							newIndividual2.evaluate(data);
						}
					}
					offspring.addIndividual(newIndividual1);
					offspring.addIndividual(newIndividual2);
				}
			}
			population = selectSurvivors(offspring);
			updateCurrentBest();
			printState();
			storeValues();
			currentGeneration++;
			stopCriteria = updateStopCriteria(getCurrentBest(), getGlobalBest());
		}
		for (int i = 0; i < population.getSize(); i++) {
			data.trainingData = originalTrainingData;
			data.unseenData = originalUnseenData;
			population.getIndividual(i).evaluate(data);
			updateCurrentBest();
			// population.getIndividual(i).evaluateOnTestData(data);
		}
		return getCurrentBest();
		// return getGlobalBest();
	}

	protected void printState() {
		if (printAtEachGeneration) {

			// System.out.println("\nGeneration:\t\t" + currentGeneration);
			// System.out.printf(
			// "Training error:\t\t%.2f\nUnseen
			// error:\t\t%.2f\nTesterror:\t\t%.2f\nSize:\t\t\t%d\nDepth:\t\t\t%d\n",
			// currentBest.getTrainingError(), currentBest.getUnseenError(),
			// currentBest.getTestError(),
			// currentBest.getSize(), currentBest.getDepth());

			System.out.println("\nGeneration:\t\t" + currentGeneration);
			System.out.printf("Trainingerror:\t\t%.2f\nUnseenerror:\t\t%.2f\nSize:\t\t\t%d\nDepth:\t\t\t%d\n",
					currentBest.getTrainingError(), currentBest.getUnseenError(), currentBest.getSize(),
					currentBest.getDepth());
		}
	}

	protected boolean updateStopCriteria(Individual currentBest, Individual globalBest) {
		// check if the current best individual is better than the best overall
		// if (currentBest.trainingError < globalBest.trainingError) {
		// runsWithoutImprovement = 0;
		// globalBest = currentBest;
		//
		if (currentBest.getAbsErrorDiff() < globalBest.getAbsErrorDiff()) {
			runsWithoutImprovement = 0;
			updateGlobalBest(currentBest);
		} else {
			runsWithoutImprovement++;
		}
		// check if maximum number of runs without improvements has been reached
		if (runsWithoutImprovement > maxRunsWithoutImprovements) {
			return true;
		} else {
			return false;
		}
	}

	// tournament selection
	protected Individual selectParent() {
		Population tournamentPopulation = new Population();
		int tournamentSize = (int) (0.05 * population.getSize());
		for (int i = 0; i < tournamentSize; i++) {
			int index = randomGenerator.nextInt(population.getSize());
			tournamentPopulation.addIndividual(population.getIndividual(index));
		}
		return tournamentPopulation.getBest();
	}

	protected Individual[] uniformCrossover(Individual p1, Individual p2) {
		ArrayList<Integer> p1SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> p2SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> SimilarSchemeIndex = new ArrayList<Integer>();
		Individual offspring[] = new Individual[2];
		offspring[0] = p1;
		offspring[1] = p2;
		SimilarSchemeIndex = calculateSimilarScheme(0, 0, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
		offspring[0] = p1.deepCopy();
		offspring[1] = p2.deepCopy();

		if (SimilarSchemeIndex.isEmpty()) {
			offspring[0] = applyStandardCrossover(p1, p2);
			offspring[1] = applyStandardCrossover(p1, p2);
		} else {
			for (int i = 0; i < p1SimilarSchemeIndex.size(); i++) {
				double randsize = Math.random();
				if (randsize < (1 / p1SimilarSchemeIndex.size())) {
					int p1CrossoverStart = p1SimilarSchemeIndex.get(i);
					int p2CrossoverStart = p2SimilarSchemeIndex.get(i);
					// If following Element in the common Scheme then only copy
					// the
					// picked Element
					if (p1SimilarSchemeIndex.contains(p1CrossoverStart + 1)) {
						int p1ElementsToEnd = 1;
						int p2ElementstoEnd = 1;
						for (int j = 0; j < p2ElementstoEnd; j++) {
							offspring[0].addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + j),
									p1CrossoverStart + j);
						}
						for (int k = 0; k < p1ElementsToEnd; k++) {
							offspring[1].addProgramElementAtIndex(p1.getProgramElementAtIndex(p1CrossoverStart + k),
									p2CrossoverStart + k);
						}
					} else {
						int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart) - 1;
						int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart) - 1;
						for (int j = 0; j < p2ElementsToEnd; j++) {
							offspring[0].addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + j),
									p1CrossoverStart + j);
						}
						for (int k = 0; k < p1ElementsToEnd; k++) {
							offspring[1].addProgramElementAtIndex(p1.getProgramElementAtIndex(p1CrossoverStart + k),
									p2CrossoverStart + k);
						}
					}
				} else {
				}
			}
		}
		offspring[0].calculateDepth();
		offspring[1].calculateDepth();
		return offspring;
	}

	protected Individual singleUniformCrossover(Individual p1, Individual p2) {
		ArrayList<Integer> p1SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> p2SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> SimilarSchemeIndex = new ArrayList<Integer>();
		Individual offspring = p1;
		SimilarSchemeIndex = calculateSimilarScheme(0, 0, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
		offspring = p1.deepCopy();
		for (int i = 0; i < p1SimilarSchemeIndex.size(); i++) {
			double randsize = Math.random();
			if (randsize < (1 / p1SimilarSchemeIndex.size())) {
				int p1CrossoverStart = p1SimilarSchemeIndex.get(i);
				int p2CrossoverStart = p2SimilarSchemeIndex.get(i);
				// If following Element in the common Scheme then only copy the
				// picked Element
				if (p1SimilarSchemeIndex.contains(p1CrossoverStart + 1)) {
					int p1ElementsToEnd = 1;
					int p2ElementstoEnd = 1;
					for (int j = 0; j < p2ElementstoEnd; j++) {
						offspring.addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + j),
								p1CrossoverStart + j);
					}
				} else {
					int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart) - 1;
					int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart) - 1;
					for (int j = 0; j < p2ElementsToEnd; j++) {
						offspring.addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + j),
								p1CrossoverStart + j);
					}
				}
			} else {
			}
		}
		offspring.calculateDepth();
		return offspring;
	}

	protected Individual[] onePointCrossover(Individual p1, Individual p2) {
		ArrayList<Integer> p1SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> p2SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> SimilarSchemeIndex = new ArrayList<Integer>();
		Individual offspring[] = new Individual[2];
		offspring[0] = p1;
		offspring[1] = p2;
		SimilarSchemeIndex = calculateSimilarScheme(0, 0, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
		if (SimilarSchemeIndex.isEmpty()) {
			offspring = applyStandardCrossoverArray(p1, p2);
		} else {
			Random r = new Random();
			int crossoverPointIndex = (r.nextInt(p1SimilarSchemeIndex.size()));
			int p1CrossoverStart = p1SimilarSchemeIndex.get(crossoverPointIndex);
			int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart);
			int p2CrossoverStart = p2SimilarSchemeIndex.get(crossoverPointIndex);
			int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart);
			offspring[0] = p1.selectiveDeepCopy(p1CrossoverStart, p1CrossoverStart + p1ElementsToEnd - 1);
			for (int i = 0; i < p2ElementsToEnd; i++) {
				offspring[0].addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + i),
						p1CrossoverStart + i);
			}
			offspring[1] = p2.selectiveDeepCopy(p2CrossoverStart, p2CrossoverStart + p2ElementsToEnd - 1);
			for (int i = 0; i < p1ElementsToEnd; i++) {
				offspring[1].addProgramElementAtIndex(p1.getProgramElementAtIndex(p1CrossoverStart + i),
						p2CrossoverStart + i);
			}
			offspring[0].calculateDepth();
			offspring[1].calculateDepth();
			return offspring;
		}
		return offspring;
	}

	protected Individual[] randomTwoOffspringsCrossover(Individual p1, Individual p2) {
		Individual offspring[] = new Individual[2];
		double randsize = Math.random();
		if (randsize > (0.9)) {
			offspring = uniformCrossover(p1, p2);
		} else if (randsize > 0.8 && randsize < 0.9) {
			offspring = onePointCrossover(p1, p2);
		} else {
			offspring = applyStandardCrossoverArray(p1, p2);
		}
		return offspring;
	}

	protected Individual randomCrossover(Individual p1, Individual p2) {
		Individual offspring;
		double randsize = Math.random();
		if (randsize > (0.75)) {
			offspring = singleUniformCrossover(p1, p2);
		} else if (randsize > 0.50 && randsize < 0.75) {
			offspring = singleOnePointCrossover(p1, p2);
		} else {
			offspring = applyStandardCrossover(p1, p2);
		}
		return offspring;
	}

	protected Individual adjustedRandomCrossover(Individual p1, Individual p2) {
		Individual offspring;
		if (currentGeneration > 300) {
			double randsize = Math.random();
			if (randsize > (0.66)) {
				offspring = singleUniformCrossover(p1, p2);
			} else if (randsize > 0.33 && randsize < 0.66) {
				offspring = singleOnePointCrossover(p1, p2);
			} else {
				offspring = applyStandardCrossover(p1, p2);
			}
		} else {
			offspring = applyStandardCrossover(p1, p2);
		}
		return offspring;
	}

	protected Individual singleOnePointCrossover(Individual p1, Individual p2) {
		ArrayList<Integer> p1SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> p2SimilarSchemeIndex = new ArrayList<Integer>();
		ArrayList<Integer> SimilarSchemeIndex = new ArrayList<Integer>();
		Individual offspring = p1;
		SimilarSchemeIndex = calculateSimilarScheme(0, 0, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
		if (SimilarSchemeIndex.isEmpty()) {
			offspring = applyStandardCrossover(p1, p2);
		} else {
			Random r = new Random();
			int crossoverPointIndex = (r.nextInt(p1SimilarSchemeIndex.size()));

			int p1CrossoverStart = p1SimilarSchemeIndex.get(crossoverPointIndex);
			int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart);
			int p2CrossoverStart = p2SimilarSchemeIndex.get(crossoverPointIndex);
			int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart);
			offspring = p1.selectiveDeepCopy(p1CrossoverStart, p1CrossoverStart + p1ElementsToEnd - 1);
			for (int i = 0; i < p2ElementsToEnd; i++) {
				offspring.addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + i),
						p1CrossoverStart + i);
			}

			offspring.calculateDepth();
			return offspring;
		}
		return offspring;
	}

	// Method to find the similar scheme
	public ArrayList<Integer> calculateSimilarScheme(int p1index, int p2index, Individual p1, Individual p2,
			ArrayList<Integer> p1SimilarSchemeIndex, ArrayList<Integer> p2SimilarSchemeIndex) {
		ArrayList<ProgramElement> p1Program = p1.getProgram();
		ArrayList<ProgramElement> p2Program = p2.getProgram();
		ArrayList<Integer> SimilarSchemeIndex = new ArrayList<Integer>();
		if (p1index < p1.getSize() & p2index < p2.getSize()) {
			if (p1Program.get(p1index) instanceof Operator & p2Program.get(p2index) instanceof Operator) {
				Operator p1currentOperator = (Operator) p1Program.get(p1index);
				Operator p2currentOperator = (Operator) p2Program.get(p2index);
				p1SimilarSchemeIndex.add(p1index);
				p2SimilarSchemeIndex.add(p2index);
				if (p1currentOperator.getArity() == p2currentOperator.getArity()) {
					p1index++;
					p2index++;
					calculateSimilarScheme(p1index, p2index, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
				} else {
					// Called if Operators don't have the same Arity
					p1index = p1index + CalculateBranchToEnd(p1index, p1);
					p2index++;
					calculateSimilarScheme(p1index, p2index, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
					p2index = p2index + CalculateBranchToEnd(p2index, p2);
					p1index++;
					calculateSimilarScheme(p1index, p2index, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
				}
			} else if ((!(p1Program.get(p1index) instanceof Operator)
					& (!(p2Program.get(p2index) instanceof Operator)))) {
				// Inner Check
				p1SimilarSchemeIndex.add(p1index);
				p2SimilarSchemeIndex.add(p2index);
				p1index++;
				p2index++;
				calculateSimilarScheme(p1index, p2index, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
			} else {
				if (p1Program.get(p1index) instanceof Operator) {
					p1index = p1index + CalculateBranchToEnd(p1index, p1);
					p2index++;
					calculateSimilarScheme(p1index, p2index, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
				} else if (p2Program.get(p2index) instanceof Operator) {
					p2index = p2index + CalculateBranchToEnd(p2index, p2);
					p1index++;
					calculateSimilarScheme(p1index, p2index, p1, p2, p1SimilarSchemeIndex, p2SimilarSchemeIndex);
				}
			}
		} else {
		}
		SimilarSchemeIndex.addAll(p1SimilarSchemeIndex);
		SimilarSchemeIndex.addAll(p2SimilarSchemeIndex);
		return (SimilarSchemeIndex);
	}

	public int CalculateBranchToEnd(int currentIndex, Individual p) {
		ArrayList<ProgramElement> program = p.getProgram();
		if (program.get(currentIndex) instanceof Terminal) {
			return 1;
		} else {
			Operator operator = (Operator) program.get(currentIndex);
			int numberOfElements = 1;
			for (int i = 0; i < operator.getArity(); i++) {
				numberOfElements += CalculateBranchToEnd(currentIndex + numberOfElements, p);
			}
			return numberOfElements;
		}
	}

	protected Individual[] adjustedStandardCrossover(Individual p1, Individual p2) {
		int p1CrossoverStart = randomGenerator.nextInt(p1.getSize());
		int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart);
		int p2CrossoverStart = randomGenerator.nextInt(p2.getSize());
		int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart);
		// add the selected tree from the second parent to the offspring
		Individual offspring[] = new Individual[2];
		offspring[0] = p1.selectiveDeepCopy(p1CrossoverStart, p1CrossoverStart + p1ElementsToEnd - 1);
		offspring[1] = p2.selectiveDeepCopy(p2CrossoverStart, p2CrossoverStart + p2ElementsToEnd - 1);

		for (int i = 0; i < p2ElementsToEnd; i++) {
			offspring[0].addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + i),
					p1CrossoverStart + i);
		}
		for (int i = 0; i < p1ElementsToEnd; i++) {
			offspring[1].addProgramElementAtIndex(p1.getProgramElementAtIndex(p1CrossoverStart + i),
					p2CrossoverStart + i);
		}

		offspring[0].calculateDepth();
		offspring[1].calculateDepth();
		return offspring;
	}

	protected Individual[] applyStandardCrossoverArray(Individual p1, Individual p2) {
		int p1CrossoverStart = randomGenerator.nextInt(p1.getSize());
		int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart);
		int p2CrossoverStart = randomGenerator.nextInt(p2.getSize());
		int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart);
		Individual offspring[] = new Individual[1];
		offspring[0] = p1.selectiveDeepCopy(p1CrossoverStart, p1CrossoverStart + p1ElementsToEnd - 1);
		// add the selected tree from the second parent to the offspring
		for (int i = 0; i < p2ElementsToEnd; i++) {
			offspring[0].addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + i),
					p1CrossoverStart + i);
		}
		offspring[0].calculateDepth();
		return offspring;
	}

	protected Individual applyStandardCrossover(Individual p1, Individual p2) {

		int p1CrossoverStart = randomGenerator.nextInt(p1.getSize());
		int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart);
		int p2CrossoverStart = randomGenerator.nextInt(p2.getSize());
		int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart);

		Individual offspring = p1.selectiveDeepCopy(p1CrossoverStart, p1CrossoverStart + p1ElementsToEnd - 1);

		// add the selected tree from the second parent to the offspring
		for (int i = 0; i < p2ElementsToEnd; i++) {
			offspring.addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + i), p1CrossoverStart + i);
		}

		offspring.calculateDepth();
		return offspring;
	}

	protected Individual applyStandardMutation(Individual p) {
		int mutationPoint = randomGenerator.nextInt(p.getSize());
		int parentElementsToEnd = p.countElementsToEnd(mutationPoint);
		Individual offspring = p.selectiveDeepCopy(mutationPoint, mutationPoint + parentElementsToEnd - 1);
		int maximumDepth = 6;
		Individual randomTree = grow(maximumDepth);
		// add the random tree to the offspring
		for (int i = 0; i < randomTree.getSize(); i++) {
			offspring.addProgramElementAtIndex(randomTree.getProgramElementAtIndex(i), mutationPoint + i);
		}
		offspring.calculateDepth();
		return offspring;
	}

	protected Individual applyNodeFlipMutation(Individual p) {
		Individual offspring = p.deepCopy();
		double nodeMutProb = 0.05;
		int curMutations = 0;
		int i = 0;
		// go over each node and mutate it with a pre-specified probability
		while (i < offspring.getSize() || curMutations <= Math.round(maxNumberOfMutations * offspring.getSize() + 5)) {
			if (randomGenerator.nextDouble() < nodeMutProb) {
				i = randomGenerator.nextInt(offspring.getSize());
				ProgramElement elementAtI = offspring.getProgramElementAtIndex(i);
				if (elementAtI instanceof InputVariable) {
					ProgramElement newNodeElement = terminalSet
							.get(randomGenerator.nextInt(terminalSet.size() - constantsLength) + constantsLength);
					offspring.setProgramElementAtIndex(newNodeElement, i);
				} else if (elementAtI instanceof Constant) {
					ProgramElement newNodeElement = terminalSet.get(randomGenerator.nextInt(constantsLength));
					offspring.setProgramElementAtIndex(newNodeElement, i);
				} else {
					Operator operator = (Operator) offspring.getProgramElementAtIndex(i);
					int ar = operator.getArity();
					Operator newOp = (Operator) functionSet.get(randomGenerator.nextInt(functionSet.size()));
					while (!(ar == newOp.getArity())) {
						newOp = (Operator) functionSet.get(randomGenerator.nextInt(functionSet.size()));
					}
					offspring.setProgramElementAtIndex(newOp, i);
				}
				curMutations++;
			}
			i++;
		}
		offspring.calculateDepth();
		return offspring;
	}

	protected Individual applyShrinkMutation(Individual p) {
		// select a random node in the tree and mutate it into a terminal
		// (deleting the subtree under that node)
		int mutationPoint = randomGenerator.nextInt(p.getSize());
		int parentElementsToEnd = p.countElementsToEnd(mutationPoint);
		Individual offspring = p.selectiveDeepCopy(mutationPoint, mutationPoint + parentElementsToEnd - 1);
		ProgramElement randTerm = terminalSet.get(randomGenerator.nextInt(terminalSet.size()));
		offspring.setProgramElementAtIndex(randTerm, mutationPoint);
		return offspring;
	}

	protected Individual applyConstantMutation(Individual p) {
		// add gaussian noise to a constant
		int mutationPoint = randomGenerator.nextInt(p.getSize());
		Individual offspring = p.deepCopy();
		ProgramElement elementAtMP = offspring.getProgramElementAtIndex(mutationPoint);
		while (!(elementAtMP instanceof Constant)) {
			mutationPoint = randomGenerator.nextInt(offspring.getSize());
			elementAtMP = offspring.getProgramElementAtIndex(mutationPoint);
		}
		Constant c = (Constant) offspring.getProgramElementAtIndex(mutationPoint);
		c.setValue(c.getValue() + randomGenerator.nextGaussian() * c.getValue());
		offspring.setProgramElementAtIndex(c, mutationPoint);
		return offspring;
	}

	// keep the best overall + all the remaining offsprings
	protected Population selectSurvivors(Population newIndividuals) {
		Population survivors = new Population();
		Individual bestParent = population.getBest();
		Individual bestNewIndividual = newIndividuals.getBest();
		Individual bestOverall;
		// the best overall is in the current population
		if (bestParent.getTrainingError() < bestNewIndividual.getTrainingError()) {
			bestOverall = bestParent;
		}
		// the best overall is in the offspring population
		else {
			bestOverall = bestNewIndividual;
		}
		survivors.addIndividual(bestOverall);
		for (int i = 0; i < newIndividuals.getSize(); i++) {
			if (newIndividuals.getIndividual(i).getId() != bestOverall.getId()) {
				survivors.addIndividual(newIndividuals.getIndividual(i));
			}
		}
		return survivors;
	}

	protected void updateCurrentBest() {
		currentBest = population.getBest();
	}

	protected void updateGlobalBest(Individual i) {
		globalBest = i;
	}

	// ##### get's and set's from here on #####
	public Individual getCurrentBest() {
		return currentBest;
	}

	public Individual getGlobalBest() {
		return globalBest;
	}

	public ArrayList<ProgramElement> getFunctionSet() {
		return functionSet;
	}

	public ArrayList<ProgramElement> getTerminalSet() {
		return terminalSet;
	}

	public ArrayList<ProgramElement> getFullSet() {
		return fullSet;
	}

	public boolean getApplyDepthLimit() {
		return applyDepthLimit;
	}

	public int getMaximumDepth() {
		return maximumDepth;
	}

	public double getCrossoverProbability() {
		return crossoverProbability;
	}

	public int getCurrentGeneration() {
		return currentGeneration;
	}

	public Data getData() {
		return data;
	}

	public Population getPopulation() {
		return population;
	}

	public int getPopulationSize() {
		return populationSize;
	}

	public Random getRandomGenerator() {
		return randomGenerator;
	}

	public boolean getPrintAtEachGeneration() {
		return printAtEachGeneration;
	}

	public void setFunctionSet(ArrayList<ProgramElement> functionSet) {
		this.functionSet = functionSet;
	}

	public void setTerminalSet(ArrayList<ProgramElement> terminalSet) {
		this.terminalSet = terminalSet;
	}

	public void setFullSet(ArrayList<ProgramElement> fullSet) {
		this.fullSet = fullSet;
	}

	public void setApplyDepthLimit(boolean applyDepthLimit) {
		this.applyDepthLimit = applyDepthLimit;
	}

	public void setMaximumDepth(int maximumDepth) {
		this.maximumDepth = maximumDepth;
	}

	public void setCrossoverProbability(double crossoverProbability) {
		this.crossoverProbability = crossoverProbability;
	}

	public void setPrintAtEachGeneration(boolean printAtEachGeneration) {
		this.printAtEachGeneration = printAtEachGeneration;
	}
}
