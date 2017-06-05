package programElements;

import utils.Utils;

public class LogisticFunction extends Operator {

	private static final long serialVersionUID = 7L;

	public LogisticFunction() {
		super(1);
	}

	public double performOperation(double... arguments) {
		return Utils.logisticFunction(arguments[0]);
	}

	public String toString() {
		return "LF";
	}
}
