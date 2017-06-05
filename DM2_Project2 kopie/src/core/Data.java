package core;

import java.io.Serializable;

public class Data implements Serializable {

	private static final long serialVersionUID = 7L;

	protected double[][] trainingData, unseenData;

	public Data(double[][] trainingData, double[][] unseenData) {
		this.trainingData = trainingData;
		this.unseenData = unseenData;
	}

	// assumes one output variable
	public int getDimensionality() {
		return trainingData[0].length - 1;
	}

	public double[][] getTrainingData() {
		return trainingData;
	}

	public double[][] getUnseenData() {
		return unseenData;
	}

	public void setTrainingData(double[][] trainingData) {
		this.trainingData = trainingData;
	}

	public void setUnseenData(double[][] unseenData) {
		this.unseenData = unseenData;
	}
}
