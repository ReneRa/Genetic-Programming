package core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import programElements.Addition;
import programElements.Constant;
import programElements.InputVariable;
import programElements.LogisticFunction;
import programElements.Multiplication;
import programElements.Operator;
import programElements.ProgramElement;
import programElements.ProtectedDivision;
import programElements.Subtraction;
import utils.Utils;

public class GpRun implements Serializable {

	private static final long serialVersionUID = 7L;

	// ##### parameters #####
	protected Data data;
	protected double[][] originalTrainingData;
	protected ArrayList<ProgramElement> functionSet, terminalSet, fullSet;
	protected int populationSize;
	protected boolean applyDepthLimit;
	protected int maximumDepth;
	protected double crossoverProbability;
	protected double mutationProbability;
	protected boolean printAtEachGeneration;
	protected double probToGrowOperator;
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
	protected double chooseSingleSubSampleProbability;

	public GpRun(Data data, boolean interleavedSampling) {
		this.data = data;
		this.originalTrainingData = data.getTrainingData();
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
		
		chooseSingleSubSampleProbability = 0.5;
		randomGenerator = new Random();
		
		if(this.interleavedSampling == true){data.trainingData = getSubSample(originalTrainingData, chooseSingleSubSampleProbability);}
		
		// adds all the constants to the terminal set
		terminalSet = new ArrayList<ProgramElement>();
		double[] constants = { -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 };
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

		populationSize = 100;
		applyDepthLimit = true;
		maximumDepth = 17;
		crossoverProbability = 0.9;
		mutationProbability = 0.1;
		probToGrowOperator = 0.5;
		printAtEachGeneration = true;
		mutationOperator = 1;
		maxNumberOfMutations = 0.05; 
		runsWithoutImprovement = 0;
		maxRunsWithoutImprovements = (int) Math.round(Main.NUMBER_OF_GENERATIONS * 0.1);

		currentGeneration = 0;

		// initialize and evaluate population
		rampedHalfAndHalfInitialization();
		for (int i = 0; i < populationSize; i++) {
			population.getIndividual(i).evaluate(data);
		}

		updateCurrentBest();
		globalBest = getCurrentBest();
		printState();
		storeValues();
		currentGeneration++;
	}
	
	// Returns a sub selection of the training data, used for cross validation
	public double[][] getSubSample(double[][] originalTrainingData, double chooseSingleSubSampleProbability) {
		if(randomGenerator.nextDouble() < chooseSingleSubSampleProbability){
			return originalTrainingData;
		}
		else{
			double[][] trainingSubSample;
			
			List<Integer> instances = Utils.shuffleInstances(originalTrainingData.length);
			int subSampleInstances = (int) Math.ceil(0.6 * originalTrainingData.length);
	
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
		 * depth at the root node is 0. this implies that the number of
		 * different depths is equal to the maximumInitialDepth
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
			Operator randomOperator = (Operator) functionSet.get(randomGenerator.nextInt(functionSet.size()));
			individual.addProgramElement(randomOperator);
			for (int i = 0; i < randomOperator.getArity(); i++) {
				fullInner(individual, currentDepth + 1, maximumTreeDepth);
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
			// run for 0.9 and 0.1
			
			if (randomGenerator.nextDouble() < probToGrowOperator) {
				Operator randomOperator = (Operator) functionSet.get(randomGenerator.nextInt(functionSet.size()));
				individual.addProgramElement(randomOperator);
				for (int i = 0; i < randomOperator.getArity(); i++) {
					growInner(individual, currentDepth + 1, maximumTreeDepth, probToGrowOperator);
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
			if(this.interleavedSampling == true){data.trainingData = getSubSample(originalTrainingData, this.chooseSingleSubSampleProbability);}
			Population offspring = new Population();
			
			// generate a new offspring population
			while (offspring.getSize() < population.getSize()) {
				Individual p1, newIndividual;
				p1 = selectParent();
				// apply crossover
				if (randomGenerator.nextDouble() < 1 - crossoverProbability - mutationProbability) {
					//	apply reproduction
						newIndividual = p1.deepCopy();
				} else { 
					if (randomGenerator.nextDouble() < crossoverProbability / (crossoverProbability + mutationProbability)) {
						Individual p2 = selectParent();
						newIndividual = applyStandardCrossover(p1, p2);
					}
					// apply mutation
					// if (randomGenerator.nextDouble() < mutationProbability) 
					else {
						if (p1.getSize() < 300000 ) {
							newIndividual = applyStandardMutation(p1);
		//					newIndividual = applyNodeFlipMutation(p1);							
						} else {
							newIndividual = applyConstantMutation(p1);
						}
						
					}
				}
				/*
				 * add the new individual to the offspring population if its
				 * depth is not higher than the maximum (applicable only if the
				 * depth limit is enabled)
				 */
				if (applyDepthLimit && newIndividual.getDepth() > maximumDepth) {
					newIndividual = p1;
				} else {
					newIndividual.evaluate(data);
				}
				offspring.addIndividual(newIndividual);
			}

			population = selectSurvivors(offspring);
			updateCurrentBest();
			printState();
			storeValues();
			currentGeneration++;
			stopCriteria = updateStopCriteria(getCurrentBest(), getGlobalBest());
		}
		return getGlobalBest();
	}

	protected void printState() {
		if (printAtEachGeneration) {
			System.out.println("\nGeneration:\t\t" + currentGeneration);
			System.out.printf("Training error:\t\t%.2f\nUnseen error:\t\t%.2f\nSize:\t\t\t%d\nDepth:\t\t\t%d\n",
					currentBest.getTrainingError(), currentBest.getUnseenError(), currentBest.getSize(),
					currentBest.getDepth());
		}
	}

	protected boolean updateStopCriteria(Individual currentBest, Individual globalBest) {
		// check if the current best individual is better than the best overall
//		if (currentBest.trainingError < globalBest.trainingError) {
//			runsWithoutImprovement = 0;
//			globalBest = currentBest;
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
			
			if (randomGenerator.nextDouble() < nodeMutProb){
				i = randomGenerator.nextInt(offspring.getSize());
				ProgramElement elementAtI = offspring.getProgramElementAtIndex(i);
				
				if (elementAtI instanceof InputVariable) {
					ProgramElement newNodeElement = terminalSet.get(randomGenerator.nextInt(terminalSet.size() - constantsLength) + constantsLength);
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
		// select a random node in the tree and mutate it into a terminal (deleting the subtree under that node)
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
		offspring.setProgramElementAtIndex((ProgramElement)c, mutationPoint); 
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
