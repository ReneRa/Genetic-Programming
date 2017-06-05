package programElements;

public class Addition extends Operator {

	private static final long serialVersionUID = 7L;

	public Addition() {
		super(2);
	}

	public double performOperation(double... arguments) {
		return arguments[0] + arguments[1];
	}

	public String toString() {
		return "+";
	}
}
