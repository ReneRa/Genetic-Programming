package programElements;

public class InputVariable extends Terminal {

	private static final long serialVersionUID = 7L;

	protected int variableIndex;

	public InputVariable(int variableIndex) {
		this.variableIndex = variableIndex;
	}

	public double getValue(double[] dataInstance) {
		return dataInstance[variableIndex];
	}

	public int getVariableIndex() {
		return variableIndex;
	}

	public String toString() {
		return "X" + variableIndex;
	}
}
