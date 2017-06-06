package core;

import java.io.Serializable;
import java.util.ArrayList;

import programElements.Constant;
import programElements.InputVariable;
import programElements.Operator;
import programElements.ProgramElement;
import programElements.Terminal;
import utils.Utils;

public class Individual implements Serializable {
	private static final long serialVersionUID = 7L;
	protected static long nextId;
	protected long id;
	protected ArrayList<ProgramElement> program;
	protected int depth;
	protected double trainingError, unseenError, testError, absDiffTrUnError;
	protected double[] trainingDataOutputs, unseenDataOutputs, testDataOutputs;
	protected int evaluateIndex;
	protected int maximumDepthAchieved;
	protected int depthCalculationIndex;
	protected int printIndex;
	protected boolean sizeOverride;
	protected int computedSize;

	public Individual() {
		program = new ArrayList<ProgramElement>();
		id = getNextId();
	}

	protected static long getNextId() {
		return nextId++;
	}

	public void evaluate(Data data) {
		evaluateOnTrainingData(data);
		evaluateOnUnseenData(data);
		evaluateOnTestData(data);
		absDiffTrUnError = Math.abs(this.unseenError - this.trainingError);
	}

	public double[] evaluateOnTrainingData(Data data) {
		double[][] trainingData = data.getTrainingData();
		if (sizeOverride == false) {
			trainingDataOutputs = evaluate(trainingData);
		}
		trainingError = Utils.calculateRMSE(trainingData, trainingDataOutputs);
		return trainingDataOutputs;
	}

	public double[] evaluateOnTestData(Data data) {
		double[][] testData = data.getTestData();
		if (sizeOverride == false) {
			testDataOutputs = evaluate(testData);
		}
		testError = Utils.calculateRMSE(testData, testDataOutputs);
		return testDataOutputs;
	}

	public double[] evaluateOnUnseenData(Data data) {
		double[][] unseenData = data.getUnseenData();
		if (sizeOverride == false) {
			unseenDataOutputs = evaluate(unseenData);
		}
		unseenError = Utils.calculateRMSE(unseenData, unseenDataOutputs);
		return unseenDataOutputs;
	}

	public double[] evaluate(double[][] data) {
		double[] outputs = new double[data.length];
		for (int i = 0; i < outputs.length; i++) {
			evaluateIndex = 0;
			outputs[i] = evaluateInner(data[i]);
		}
		return outputs;
	}

	protected double evaluateInner(double[] dataInstance) {
		if (program.get(evaluateIndex) instanceof InputVariable) {
			InputVariable inputVariable = (InputVariable) program.get(evaluateIndex);
			return inputVariable.getValue(dataInstance);
		} else if (program.get(evaluateIndex) instanceof Constant) {
			Constant constant = (Constant) program.get(evaluateIndex);
			return constant.getValue();
		} else {
			Operator operator = (Operator) program.get(evaluateIndex);
			double[] arguments = new double[operator.getArity()];
			for (int i = 0; i < arguments.length; i++) {
				evaluateIndex++;
				arguments[i] = evaluateInner(dataInstance);
			}
			return operator.performOperation(arguments);
		}
	}

	public Individual deepCopy() {
		Individual newIndividual = new Individual();
		for (int i = 0; i < program.size(); i++) {
			newIndividual.program.add(program.get(i));
		}
		newIndividual.setDepth(depth);
		return newIndividual;
	}

	// The resulting copy is: [0, exclusionZoneStart[ + ]exclusionZoneEnd, N-1]
	public Individual selectiveDeepCopy(int exclusionZoneStartIndex, int exclusionZoneEndIndex) {
		Individual newIndividual = new Individual();
		for (int i = 0; i < exclusionZoneStartIndex; i++) {
			newIndividual.program.add(program.get(i));
		}
		for (int i = exclusionZoneEndIndex + 1; i < program.size(); i++) {
			newIndividual.program.add(program.get(i));
		}
		return newIndividual;
	}

	public void calculateDepth() {
		maximumDepthAchieved = 0;
		depthCalculationIndex = 0;
		calculateDepth(0);
		depth = maximumDepthAchieved;
	}

	protected void calculateDepth(int currentDepth) {
		if (program.get(depthCalculationIndex) instanceof Operator) {
			Operator currentOperator = (Operator) program.get(depthCalculationIndex);
			for (int i = 0; i < currentOperator.getArity(); i++) {
				depthCalculationIndex++;
				calculateDepth(currentDepth + 1);
			}
		} else {
			if (currentDepth > maximumDepthAchieved) {
				maximumDepthAchieved = currentDepth;
			}
		}
	}

	public int countElementsToEnd(int startingIndex) {
		if (program.get(startingIndex) instanceof Terminal) {
			return 1;
		} else {
			Operator operator = (Operator) program.get(startingIndex);
			int numberOfElements = 1;
			for (int i = 0; i < operator.getArity(); i++) {
				numberOfElements += countElementsToEnd(startingIndex + numberOfElements);
			}
			return numberOfElements;
		}
	}

	public void addProgramElement(ProgramElement programElement) {
		program.add(programElement);
	}

	public void addProgramElementAtIndex(ProgramElement programElement, int index) {
		program.add(index, programElement);
	}

	public void removeProgramElementAtIndex(int index) {
		program.remove(index);
	}

	public ProgramElement getProgramElementAtIndex(int index) {
		return program.get(index);
	}

	public void setProgramElementAtIndex(ProgramElement programElement, int index) {
		program.set(index, programElement);
	}

	public void print() {
		if (sizeOverride == true) {
			System.out.println(" [Individual not constructed]");
		} else {
			printIndex = 0;
			printInner();
		}
	}

	protected void printInner() {
		if (program.get(printIndex) instanceof Terminal) {
			System.out.print(" " + program.get(printIndex));
		} else {
			System.out.print(" (");
			System.out.print(program.get(printIndex));
			Operator currentOperator = (Operator) program.get(printIndex);
			for (int i = 0; i < currentOperator.getArity(); i++) {
				printIndex++;
				printInner();
			}
			System.out.print(")");
		}
	}

	public void writeToObjectFile() {
		if (sizeOverride == true) {
			System.out.println("Individual not constructed!");
		} else {
			Utils.writeObjectToFile(this, "final_individual.obj");
		}
	}

	// ##### get's and set's from here on #####
	public double getTrainingError() {
		return trainingError;
	}

	public double getUnseenError() {
		return unseenError;
	}

	public double getTestError() {
		return testError;
	}

	public double getAbsErrorDiff() {
		return absDiffTrUnError;
	}

	public double[] getTrainingDataOutputs() {
		return trainingDataOutputs;
	}

	public double[] getUnseenDataOutputs() {
		return unseenDataOutputs;
	}

	public long getId() {
		return id;
	}

	public int getSize() {
		if (sizeOverride) {
			return computedSize;
		} else {
			return program.size();
		}
	}

	public int getDepth() {
		return depth;
	}

	public ArrayList<ProgramElement> getProgram() {
		return program;
	}

	public void setSizeOverride(boolean sizeOverride) {
		this.sizeOverride = sizeOverride;
	}

	public void setComputedSize(int computedSize) {
		this.computedSize = computedSize;
	}

	public void setDepth(int depth) {
		this.depth = depth;
	}

	public void setTrainingDataOutputs(double[] trainingDataOutputs) {
		this.trainingDataOutputs = trainingDataOutputs;
	}

	public void setUnseenDataOutputs(double[] unseenDataOutputs) {
		this.unseenDataOutputs = unseenDataOutputs;
	}
}